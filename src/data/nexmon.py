"""
[file]          nexmon.py
[description]   Reader for nexmon CSI pcap captures (nexmon.org/csi).

                Each UDP packet in the capture holds one CSI frame for one core / spatial stream. The
                payload starts with either the 4-byte magic 0x11111111 (older firmware) or the 2-byte
                magic 0x1111 followed by RSSI and frame-control bytes (newer firmware); both are
                followed by a 6-byte source MAC, a 2-byte sequence number, a 2-byte core/spatial-stream
                field, a 2-byte chanspec, a 2-byte chip version and then the CSI itself. The CSI is 4
                bytes per subcarrier: interleaved int16 real/imag on the bcm4339 and bcm43455c0, and a
                packed sign/mantissa/exponent format on the bcm4358 and bcm4366c0.

                Only the int16 format is decoded here. The packed format needs the acphy unpacker from
                nexmon's utils/matlab/unpack_float.c and is not implemented.
"""
#
##

import struct

import numpy as np

#
## chip versions whose CSI is interleaved int16 real/imag
INT16_CHIPS = {0x0065}  # bcm43455c0 (the chip in the current captures)

PCAP_MAGIC_LE = 0xA1B2C3D4
PCAP_MAGIC_BE = 0xD4C3B2A1
LINKTYPE_ETHERNET = 1


def _parse_pcap_header(var_bytes):
    """
    [description]
    : parse the 24-byte pcap global header and return the byte order and link type.
    """
    var_magic = struct.unpack("<I", var_bytes[:4])[0]
    if var_magic == PCAP_MAGIC_LE:
        var_endian = "<"
    elif var_magic == PCAP_MAGIC_BE:
        var_endian = ">"
    else:
        raise ValueError(f"not a pcap file (magic 0x{var_magic:08x})")
    _, _, _, _, _, var_linktype = struct.unpack(var_endian + "HHiIII", var_bytes[4:24])
    return var_endian, var_linktype


def _udp_payload(var_frame):
    """
    [description]
    : strip the Ethernet / IPv4 / UDP headers off a captured frame and return the UDP payload.
    """
    if len(var_frame) < 14:
        return None
    if struct.unpack(">H", var_frame[12:14])[0] != 0x0800:
        return None
    var_ip = var_frame[14:]
    if len(var_ip) < 20 or (var_ip[0] >> 4) != 4:
        return None
    var_ihl = (var_ip[0] & 0x0F) * 4
    if var_ip[9] != 17:  # UDP
        return None
    return var_ip[var_ihl:][8:]


def _nexmon_header(var_payload):
    """
    [description]
    : parse the nexmon CSI payload header.
    [return]
    : (csi_bytes, meta) or None when the payload is not a nexmon CSI payload.
    """
    if var_payload[:4] == b"\x11\x11\x11\x11":
        ## older firmware: 4-byte magic, no RSSI/frame-control
        var_off = 4
        var_meta = {}
    elif var_payload[:2] == b"\x11\x11":
        ## newer firmware: 2-byte magic, then RSSI and frame control
        var_off = 4
        var_meta = {"rssi": struct.unpack("<b", var_payload[2:3])[0],
                    "fc": var_payload[3]}
    else:
        return None
    if len(var_payload) < var_off + 14:
        return None
    ## Layout after the magic: mac(6), seq(2), core/ss(2), chanspec(2), chip version(2)
    var_meta["mac"] = var_payload[var_off:var_off + 6].hex(":")
    var_seq, var_css, var_chanspec, var_chip = struct.unpack(
        "<HHHH", var_payload[var_off + 6:var_off + 14])
    var_meta.update({"seq": var_seq, "core_ss": var_css, "core": var_css & 0x7,
                     "spatial_stream": (var_css >> 3) & 0x7,
                     "chanspec": var_chanspec, "chip": var_chip})
    return var_payload[var_off + 14:], var_meta


def read_nexmon_pcap(var_path, var_max_frames=None):
    """
    [description]
    : read a nexmon CSI pcap capture and return the CSI amplitude of one core / spatial stream.
    : var_path: str, path of the pcap file
    : var_max_frames: int or None, stop after this many CSI frames
    : return: (amplitude (num_frames, num_subcarriers) float32, meta dict). meta holds the parsed
      header fields of the first frame, the number of subcarriers, and the counts of every
      (core, spatial_stream) and chip version seen.
    """
    #
    var_frames = []
    var_metas = []
    with open(var_path, "rb") as var_file:
        var_endian, var_linktype = _parse_pcap_header(var_file.read(24))
        if var_linktype != LINKTYPE_ETHERNET:
            raise ValueError(f"unsupported pcap link type {var_linktype} (expected Ethernet)")
        while var_max_frames is None or len(var_frames) < var_max_frames:
            var_rec = var_file.read(16)
            if len(var_rec) < 16:
                break
            _, _, var_incl, _ = struct.unpack(var_endian + "IIII", var_rec)
            var_frame = var_file.read(var_incl)
            #
            var_payload = _udp_payload(var_frame)
            if var_payload is None:
                continue
            var_parsed = _nexmon_header(var_payload)
            if var_parsed is None:
                continue
            var_csi_bytes, var_meta = var_parsed
            var_metas.append(var_meta)
            var_frames.append(var_csi_bytes)
    #
    if not var_frames:
        raise ValueError(f"no nexmon CSI packets found in {var_path}")
    #
    var_chips = {var_meta["chip"] for var_meta in var_metas}
    var_chip = var_metas[0]["chip"]
    if var_chip not in INT16_CHIPS:
        raise NotImplementedError(
            f"chip version 0x{var_chip:04x} does not use the int16 CSI format; the packed "
            f"sign/mantissa/exponent format (bcm4358/bcm4366c0) is not implemented. Chips seen: "
            + ", ".join(f"0x{var_c:04x}" for var_c in sorted(var_chips)))
    #
    ## one CSI value is 4 bytes (int16 real + int16 imag)
    var_num_subcarriers = len(var_frames[0]) // 4
    var_amplitude = np.zeros((len(var_frames), var_num_subcarriers), dtype=np.float32)
    for var_idx, var_csi_bytes in enumerate(var_frames):
        var_iq = np.frombuffer(var_csi_bytes, dtype="<i2", count=2 * var_num_subcarriers)
        var_iq = var_iq.reshape(var_num_subcarriers, 2).astype(np.float32)
        var_amplitude[var_idx] = np.sqrt(var_iq[:, 0] ** 2 + var_iq[:, 1] ** 2)
    #
    var_css_counts = {}
    for var_meta in var_metas:
        var_key = f"core{var_meta['core']}_ss{var_meta['spatial_stream']}"
        var_css_counts[var_key] = var_css_counts.get(var_key, 0) + 1
    var_meta_out = dict(var_metas[0])
    var_meta_out.update({"num_frames": len(var_frames), "num_subcarriers": var_num_subcarriers,
                         "core_spatial_stream_counts": var_css_counts,
                         "chip_versions": sorted(var_chips)})
    return var_amplitude, var_meta_out


def build_model_input(var_amplitude, var_x_shape, var_scale=None):
    """
    [description]
    : turn a captured amplitude sequence into the (time, feature) tensor the trained density model
      expects. The capture's subcarrier axis is interpolated onto the model's feature axis and the
      time axis is zero-padded on the left (the same convention load_data_x uses). This is a best
      effort transfer: the WiMANS model was trained on a different NIC, bandwidth and subcarrier
      count, so the numbers are not physically comparable, only shape-compatible.
    : var_amplitude: numpy array (num_frames, num_subcarriers) float32
    : var_x_shape: tuple (length, num_features) of the trained model
    : var_scale: float or None, optional multiplier applied to the amplitude
    : return: numpy array (length, num_features) float32
    """
    #
    var_length, var_num_features = int(var_x_shape[0]), int(var_x_shape[1])
    var_num_subcarriers = var_amplitude.shape[1]
    #
    var_source = np.arange(var_num_subcarriers, dtype=np.float64)
    var_target = np.linspace(0, var_num_subcarriers - 1, var_num_features)
    var_resampled = np.empty((var_amplitude.shape[0], var_num_features), dtype=np.float32)
    for var_idx in range(var_amplitude.shape[0]):
        var_resampled[var_idx] = np.interp(var_target, var_source, var_amplitude[var_idx])
    #
    if var_scale is not None:
        var_resampled = var_resampled * var_scale
    #
    var_input = np.zeros((var_length, var_num_features), dtype=np.float32)
    var_num_keep = min(var_length, var_resampled.shape[0])
    var_input[var_length - var_num_keep:] = var_resampled[-var_num_keep:]
    return var_input
