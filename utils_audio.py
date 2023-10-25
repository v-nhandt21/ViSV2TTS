import os
from moviepy.editor import VideoFileClip
from pydub import AudioSegment

def convert_to_wav(input_file):
    _, extension = os.path.splitext(input_file)
    extension = extension.lower()  # Convert to lowercase for case-insensitivity
    output_wav_file = input_file.replace(extension, ".wav")
    if extension == ".wav":
        return output_wav_file
    if extension == ".mp4":
        video_clip = VideoFileClip(input_file)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(output_wav_file)
        audio_clip.close()
        print(f"{input_file} (MP4) converted to {output_wav_file}")
        return output_wav_file
    elif extension == ".mp3":
        audio_clip = AudioSegment.from_mp3(input_file)
        audio_clip.export(output_wav_file, format="wav")
        print(f"{input_file} (MP3) converted to {output_wav_file}")
        return output_wav_file
    else:
        print(f"Unsupported file format: {extension}")
        return input_file