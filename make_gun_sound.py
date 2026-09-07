import wave
import random
import struct

sample_rate = 44100
duration = 0.4 # seconds
num_samples = int(sample_rate * duration)

with wave.open("gun.wav", "w") as wav_file:
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(sample_rate)
    
    for i in range(num_samples):
        # White noise
        noise = random.uniform(-1.0, 1.0)
        
        # Add some low frequency rumble
        # using a simple oscillator would be better, but noise + decay works well for a gunshot
        
        # Exponential decay envelope for that sharp "bang" sound
        decay = (1.0 - (i / num_samples)) ** 7
        
        sample = int(noise * decay * 32767 * 0.8) # 0.8 to avoid clipping
        wav_file.writeframes(struct.pack('h', sample))
