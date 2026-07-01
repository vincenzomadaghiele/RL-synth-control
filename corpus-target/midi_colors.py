import mido
import time
import random

if __name__ == "__main__":

    # MIDI INPUT
    MIDI_device_name = "Midi Fighter Twister" # substitute with the name of your MIDI device
    default_input = mido.open_input(MIDI_device_name)
    midi_port_output = mido.open_output(MIDI_device_name)
    def initializeFitherTwister(controller):
        orange = 70
        green = 50
        purple = 100
        blue = 10
        black = 0
        for j in range(64):
            if j in [0, 16, 32]:
                midi_port_output.send(mido.Message('control_change', channel=1, control=j, value=green, time=0))
            elif j in [8, 9]:
                midi_port_output.send(mido.Message('control_change', channel=1, control=j, value=purple, time=0))
            elif j in [24, 25, 26, 27]:
                midi_port_output.send(mido.Message('control_change', channel=1, control=j, value=blue, time=0))
            elif j in list(range(39,39+9)):
                midi_port_output.send(mido.Message('control_change', channel=1, control=j, value=orange, time=0))
            elif j in list(range(48,48+8)):
                midi_port_output.send(mido.Message('control_change', channel=1, control=j, value=orange, time=0))
            elif j in list(range(56,56+4)):
                midi_port_output.send(mido.Message('control_change', channel=1, control=j, value=blue, time=0))
            else:
                midi_port_output.send(mido.Message('control_change', channel=1, control=j, value=black, time=0))

