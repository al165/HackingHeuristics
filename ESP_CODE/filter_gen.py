#!/usr/bin/env python3

usage_guide = """\
This script generates C/C++ code for a digital IIR filter using the SciPy
signal processing library.  It uses the well-known Butterworth design which
optimizes flat frequency response.

The script arguments specify the filter characteristics:

The filter type determines the range of frequencies to block.  Lowpass smooths
signals by blocking high frequencies; highpass removes constant components by
blocking low frequencies; bandpass blocks both low and high frequencies to
emphasize a range of interest; bandstop blocks a range of frequencies to remove
specific unwanted frequency components such as periodic noise sources.

The sampling frequency is the constant rate at which the sensor signal is
sampled and is specified in samples per second.

The filter order determines both the number of state variables and steepness of
frequency response.  It specifies the number of terms in the frequency-space
polynomials which define the filter.

For lowpass and highpass filters, the critical frequency specifies the corner of
the idealized filter curve in Hz.  The filter has a rolloff, so the blocking
strength increases as frequencies move beyond this corner into the blocked range.

For bandpass and bandstop filters, the critical frequency is the center of the
band, and the width is the total width of the band.

The optional plot shows the magnitude of the response versus frequency.  This is
unconventional, but avoids the potential confusion of dB or a logarithmic
vertical scale.

References:
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
  https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html
  https://en.wikipedia.org/wiki/Butterworth_filter
  https://en.wikipedia.org/wiki/Digital_filter#Direct_form_II

"""


################################################################
# Standard Python libraries.
import sys, argparse, logging

# Set up debugging output.
# logging.getLogger().setLevel(logging.DEBUG)

# Extension libraries.
import numpy as np
import scipy.signal

type_print_form = {'lowpass' : 'Low-Pass', 'highpass' : 'High-Pass', 'bandpass' : 'Band-Pass', 'bandstop' : 'Band-Stop'}

################################################################
# Optionally generate plots of the filter properties.
# Refs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfreqz.html
def make_plots(filename, sos, fs, order, freqs, name):
    try:
        import matplotlib.pyplot as plt
    except:
        print("Warning, matplotlib not found, skipping plot generation.")
        return
            
    # N.B. response is a vector of complex numbers
    freq, response = scipy.signal.sosfreqz(sos, fs=fs)
    fig, ax = plt.subplots(nrows=1)
    fig.set_dpi(160)
    fig.set_size_inches((8,6))
    ax.plot(freq, np.abs(response))
    ax.set_title(f"Response of {freqs} Hz {name} Filter of Order {order}")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude of Transfer Ratio")
    fig.savefig(filename)


################################################################
def emit_biquad_code(stream, coeff):
    # Emit C code for a single biquad section in direct form II. For simplicity, the filter state
    # is embedded directly in the block as static variables, and the coefficients appear
    # directly in the expressions.
    
    # b0, b1, b2 are the numerator coefficients
    # a0, a1, a2 are the denominator coefficients
    # coefficients are normalized so a0==1
    b0, b1, b2, a0, a1, a2 = coeff
    stream.write(f"""\
  {{
    static float z1, z2; // filter section state
    float x = output - {a1:.8f}*z1 - {a2:.8f}*z2;
    output = {b0:.8f}*x + {b1:.8f}*z1 + {b2:.8f}*z2;
    z2 = z1;
    z1 = x;
  }}
""")

################################################################    
def emit_filter_function(stream, name, sos):
    stream.write(f"""\
float {name}(float input)
{{
  float output = input;
""")
    for section in sos:
        emit_biquad_code(stream, section)

    stream.write("  return output;\n}\n")

################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Generate C code for Butterworth IIR digital filters.""",
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=usage_guide)
    parser.add_argument('--type',  default='lowpass',  type=str,
                        choices = ['lowpass', 'highpass', 'bandpass', 'bandstop'], 
                        help = 'Filter type: lowpass, highpass, bandpass, bandstop (default lowpass).')
    
    parser.add_argument('--rate',  default=10,  type=float, help = 'Sampling frequency in Hz (default 10).')
    parser.add_argument('--order', default=4,   type=int,   help = 'Filter order (default 4).')
    parser.add_argument('--freq',  default=1.0, type=float, help = 'Critical frequency (default 1.0 Hz).')
    parser.add_argument('--width', default=1.0, type=float, help = 'Bandwidth (for bandpass or bandstop) (default 1.0 Hz).')
    parser.add_argument('--name',  type=str, help = 'Name of C filter function.')
    parser.add_argument('--out',   type=str, help='Path of C output file for filter code.')
    parser.add_argument('--plot',  type=str, help='Path of plot output image file.')
    args = parser.parse_args()

    if args.type == 'lowpass':
        freqs = args.freq
        funcname = 'lowpass' if args.name is None else args.name
        
    elif args.type == 'highpass':
        freqs = args.freq
        funcname = 'highpass' if args.name is None else args.name
     
    elif args.type == 'bandpass':
        freqs = [args.freq - 0.5*args.width, args.freq + 0.5*args.width]
        funcname = 'bandpass' if args.name is None else args.name
        
    elif args.type == 'bandstop':                
        freqs = [args.freq - 0.5*args.width, args.freq + 0.5*args.width]
        funcname = 'bandstop' if args.name is None else args.name    

    # Generate a Butterworth filter as a cascaded series of second-order digital
    # filters (second-order sections aka biquad).
    sos = scipy.signal.butter(N=args.order, Wn=freqs, btype=args.type, analog=False, output='sos', fs=args.rate)

    logging.debug("SOS filter: %s", sos)

    filename = args.type + '.ino' if args.out is None else args.out
    stream = open(filename, "w")

    printable_type = type_print_form[args.type]
    stream.write(f"// {printable_type} Butterworth IIR digital filter, generated using filter_gen.py.\n")
    stream.write(f"// Sampling rate: {args.rate} Hz, frequency: {freqs} Hz.\n")
    stream.write(f"// Filter is order {args.order}, implemented as second-order sections (biquads).\n")
    stream.write("// Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html\n")
    funcname = args.type if args.name is None else args.name    
    emit_filter_function(stream, funcname, sos)
    stream.close()

    if args.plot is not None:
        make_plots(args.plot, sos, args.rate, args.order, freqs, printable_type)
        
################################################################
