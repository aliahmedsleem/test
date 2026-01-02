#!/usr/bin/env python3
"""QPSK modulation demo with Arduino-compatible serial streaming.

This script:
- Reads incoming IQ samples from two serial COM ports (I & Q).
- Sends a user-typed message as QPSK-modulated samples.
- Displays an oscilloscope-like plot of the transmitted signal.
- Reports bitrate and bit error rate (BER).
- Allows adjusting carrier frequency and waveform type at runtime.

Typical usage:
  python qpsk_arduino.py --rx-port-i COM4 --rx-port-q COM5 --tx-port COM6

If no serial ports are provided, it runs a local simulation with noise.
"""
from __future__ import annotations

import argparse
import queue
import threading
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np

try:
    import serial
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(
        "pyserial is required. Install with: pip install pyserial"
    ) from exc

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(
        "matplotlib is required. Install with: pip install matplotlib"
    ) from exc


@dataclass
class ModulationParams:
    carrier_hz: float
    bit_rate: float
    sample_rate: float
    waveform: str

    @property
    def samples_per_symbol(self) -> int:
        symbol_rate = self.bit_rate / 2.0
        return max(2, int(self.sample_rate / symbol_rate))


def text_to_bits(text: str) -> List[int]:
    data = text.encode("utf-8")
    bits = []
    for byte in data:
        bits.extend([(byte >> shift) & 1 for shift in range(7, -1, -1)])
    return bits


def bits_to_text(bits: Iterable[int]) -> str:
    byte_vals = []
    bit_list = list(bits)
    for idx in range(0, len(bit_list), 8):
        chunk = bit_list[idx : idx + 8]
        if len(chunk) < 8:
            break
        value = 0
        for bit in chunk:
            value = (value << 1) | int(bit)
        byte_vals.append(value)
    return bytes(byte_vals).decode("utf-8", errors="replace")


def qpsk_map(bits: Iterable[int]) -> np.ndarray:
    bit_pairs = list(bits)
    if len(bit_pairs) % 2:
        bit_pairs.append(0)
    symbols = []
    for i in range(0, len(bit_pairs), 2):
        b1, b2 = bit_pairs[i], bit_pairs[i + 1]
        if (b1, b2) == (0, 0):
            symbols.append(1 + 1j)
        elif (b1, b2) == (0, 1):
            symbols.append(-1 + 1j)
        elif (b1, b2) == (1, 1):
            symbols.append(-1 - 1j)
        else:
            symbols.append(1 - 1j)
    return np.array(symbols, dtype=np.complex64)


def qpsk_demod(symbols: np.ndarray) -> List[int]:
    bits: List[int] = []
    for sym in symbols:
        i_val, q_val = sym.real, sym.imag
        if i_val >= 0 and q_val >= 0:
            bits.extend([0, 0])
        elif i_val < 0 <= q_val:
            bits.extend([0, 1])
        elif i_val < 0 and q_val < 0:
            bits.extend([1, 1])
        else:
            bits.extend([1, 0])
    return bits


def modulate_qpsk(
    bits: Iterable[int],
    params: ModulationParams,
    noise_std: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    symbols = qpsk_map(bits)
    sps = params.samples_per_symbol
    num_samples = len(symbols) * sps
    t = np.arange(num_samples) / params.sample_rate

    i_signal = np.repeat(symbols.real, sps)
    q_signal = np.repeat(symbols.imag, sps)

    omega = 2 * np.pi * params.carrier_hz
    if params.waveform == "cos":
        carrier_i = np.cos(omega * t)
        carrier_q = -np.sin(omega * t)
    else:
        carrier_i = np.sin(omega * t)
        carrier_q = np.cos(omega * t)

    passband = i_signal * carrier_i + q_signal * carrier_q

    if noise_std > 0:
        passband = passband + np.random.normal(0.0, noise_std, size=passband.shape)

    return t, passband.astype(np.float32), symbols


def demodulate_qpsk(
    signal: np.ndarray, params: ModulationParams
) -> np.ndarray:
    sps = params.samples_per_symbol
    t = np.arange(len(signal)) / params.sample_rate
    omega = 2 * np.pi * params.carrier_hz
    if params.waveform == "cos":
        carrier_i = np.cos(omega * t)
        carrier_q = -np.sin(omega * t)
    else:
        carrier_i = np.sin(omega * t)
        carrier_q = np.cos(omega * t)

    mixed_i = signal * carrier_i
    mixed_q = signal * carrier_q

    symbols = []
    for idx in range(0, len(signal), sps):
        i_chunk = mixed_i[idx : idx + sps]
        q_chunk = mixed_q[idx : idx + sps]
        if len(i_chunk) < sps:
            break
        symbols.append(complex(np.mean(i_chunk), np.mean(q_chunk)))
    return np.array(symbols, dtype=np.complex64)


def calc_ber(tx_bits: Iterable[int], rx_bits: Iterable[int]) -> float:
    tx_list = list(tx_bits)
    rx_list = list(rx_bits)
    if not tx_list or not rx_list:
        return 0.0
    min_len = min(len(tx_list), len(rx_list))
    errors = sum(1 for i in range(min_len) if tx_list[i] != rx_list[i])
    return errors / float(min_len)


class SerialReader(threading.Thread):
    def __init__(self, port: str, baud: int, output: queue.Queue):
        super().__init__(daemon=True)
        self._port = port
        self._baud = baud
        self._output = output
        self._stop_event = threading.Event()
        self._ser: Optional[serial.Serial] = None

    def run(self) -> None:
        self._ser = serial.Serial(self._port, self._baud, timeout=0.1)
        while not self._stop_event.is_set():
            line = self._ser.readline().decode(errors="ignore").strip()
            if not line:
                continue
            try:
                self._output.put(float(line))
            except ValueError:
                continue

    def stop(self) -> None:
        self._stop_event.set()
        if self._ser and self._ser.is_open:
            self._ser.close()


class SerialWriter:
    def __init__(self, port: str, baud: int):
        self._ser = serial.Serial(port, baud, timeout=0.1)

    def write_line(self, value: str) -> None:
        self._ser.write((value + "\n").encode("utf-8"))

    def close(self) -> None:
        if self._ser.is_open:
            self._ser.close()


def stream_iq_to_ports(
    i_samples: np.ndarray,
    q_samples: np.ndarray,
    writer_i: Optional[SerialWriter],
    writer_q: Optional[SerialWriter],
) -> None:
    if not writer_i or not writer_q:
        return
    for i_val, q_val in zip(i_samples, q_samples):
        writer_i.write_line(f"{i_val:.5f}")
        writer_q.write_line(f"{q_val:.5f}")


def parse_runtime_command(text: str, params: ModulationParams) -> bool:
    parts = text.strip().split()
    if not parts:
        return False
    cmd = parts[0].lower()
    if cmd in {"/quit", "/exit"}:
        raise KeyboardInterrupt
    if cmd == "/carrier" and len(parts) == 2:
        params.carrier_hz = float(parts[1])
        return True
    if cmd == "/bitrate" and len(parts) == 2:
        params.bit_rate = float(parts[1])
        return True
    if cmd == "/waveform" and len(parts) == 2:
        params.waveform = parts[1].lower()
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="QPSK modulation with Arduino serial IO")
    parser.add_argument("--rx-port-i", help="Serial port for I samples")
    parser.add_argument("--rx-port-q", help="Serial port for Q samples")
    parser.add_argument("--tx-port", help="Serial port for transmitting text")
    parser.add_argument("--tx-port-i", help="Serial port for I samples output")
    parser.add_argument("--tx-port-q", help="Serial port for Q samples output")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--carrier-hz", type=float, default=2000.0)
    parser.add_argument("--bit-rate", type=float, default=1000.0)
    parser.add_argument("--sample-rate", type=float, default=48000.0)
    parser.add_argument("--waveform", choices=["cos", "sin"], default="cos")
    parser.add_argument("--noise-std", type=float, default=0.2)
    args = parser.parse_args()

    params = ModulationParams(
        carrier_hz=args.carrier_hz,
        bit_rate=args.bit_rate,
        sample_rate=args.sample_rate,
        waveform=args.waveform,
    )

    rx_queue_i: queue.Queue = queue.Queue()
    rx_queue_q: queue.Queue = queue.Queue()
    reader_i: Optional[SerialReader] = None
    reader_q: Optional[SerialReader] = None

    if args.rx_port_i:
        reader_i = SerialReader(args.rx_port_i, args.baud, rx_queue_i)
        reader_i.start()
    if args.rx_port_q:
        reader_q = SerialReader(args.rx_port_q, args.baud, rx_queue_q)
        reader_q.start()

    tx_writer: Optional[SerialWriter] = None
    if args.tx_port:
        tx_writer = SerialWriter(args.tx_port, args.baud)

    tx_writer_i = SerialWriter(args.tx_port_i, args.baud) if args.tx_port_i else None
    tx_writer_q = SerialWriter(args.tx_port_q, args.baud) if args.tx_port_q else None

    fig, ax = plt.subplots()
    ax.set_title("QPSK Oscilloscope")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    line, = ax.plot([], [], lw=1)
    ax.set_ylim(-2.5, 2.5)
    ax.set_xlim(0, int(params.sample_rate / 10))

    signal_buffer: List[float] = []

    def update_plot(frame):
        window = int(params.sample_rate / 10)
        if len(signal_buffer) > window:
            data = signal_buffer[-window:]
        else:
            data = signal_buffer
        line.set_data(range(len(data)), data)
        ax.set_xlim(0, max(window, len(data)))
        return line,

    animation = FuncAnimation(fig, update_plot, interval=50)

    def collect_rx_samples() -> Optional[np.ndarray]:
        if not reader_i or not reader_q:
            return None
        i_samples = []
        q_samples = []
        while not rx_queue_i.empty() and not rx_queue_q.empty():
            i_samples.append(rx_queue_i.get())
            q_samples.append(rx_queue_q.get())
        if not i_samples:
            return None
        return np.array(i_samples, dtype=np.float32), np.array(q_samples, dtype=np.float32)

    print("Type a message to transmit. Commands: /carrier HZ, /bitrate BPS, /waveform cos|sin, /quit")

    try:
        while True:
            message = input("> ").strip()
            if not message:
                continue
            if parse_runtime_command(message, params):
                print(
                    f"Updated: carrier={params.carrier_hz}Hz bitrate={params.bit_rate} waveform={params.waveform}"
                )
                continue

            bits = text_to_bits(message)
            t, signal, symbols = modulate_qpsk(bits, params, noise_std=args.noise_std)
            signal_buffer.extend(signal.tolist())

            if tx_writer:
                tx_writer.write_line(message)

            i_signal = np.repeat(symbols.real, params.samples_per_symbol)
            q_signal = np.repeat(symbols.imag, params.samples_per_symbol)
            stream_iq_to_ports(i_signal, q_signal, tx_writer_i, tx_writer_q)

            rx_samples = collect_rx_samples()
            if rx_samples is None:
                demod_symbols = demodulate_qpsk(signal, params)
            else:
                i_rx, q_rx = rx_samples
                rx_complex = i_rx + 1j * q_rx
                demod_symbols = rx_complex

            rx_bits = qpsk_demod(demod_symbols)
            rx_message = bits_to_text(rx_bits)
            ber = calc_ber(bits, rx_bits)

            print(
                f"Sent bits: {len(bits)} | Bitrate: {params.bit_rate} bps | BER: {ber:.4f}"
            )
            print(f"Received (decoded): {rx_message}")
            plt.pause(0.01)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        if reader_i:
            reader_i.stop()
        if reader_q:
            reader_q.stop()
        if tx_writer:
            tx_writer.close()
        if tx_writer_i:
            tx_writer_i.close()
        if tx_writer_q:
            tx_writer_q.close()


if __name__ == "__main__":
    main()
