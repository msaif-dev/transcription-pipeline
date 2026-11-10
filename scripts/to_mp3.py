"""Convert an audio file to MP3 with PyAV (no system ffmpeg needed). Used to build format test samples.

    python scripts/to_mp3.py samples/long_talk.wav samples/long_talk.mp3
"""

import sys

import av


def main(src: str, dst: str, bitrate: int = 64_000) -> None:
    with av.open(src) as inp, av.open(dst, "w") as out:
        in_stream = inp.streams.audio[0]
        out_stream = out.add_stream("mp3", rate=in_stream.codec_context.sample_rate)
        out_stream.bit_rate = bitrate
        out_stream.layout = "mono"
        resampler = av.AudioResampler(format=out_stream.codec_context.format.name, layout="mono",
                                      rate=out_stream.codec_context.sample_rate)
        for frame in inp.decode(in_stream):
            frame.pts = None
            for f in resampler.resample(frame):
                for pkt in out_stream.encode(f):
                    out.mux(pkt)
        for f in resampler.resample(None):
            for pkt in out_stream.encode(f):
                out.mux(pkt)
        for pkt in out_stream.encode(None):
            out.mux(pkt)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
