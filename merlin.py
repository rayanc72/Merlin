import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from scipy.ndimage.filters import gaussian_filter
import lmfit




def preprocessing():
    import pandas as pd
    try:
        file1 = open('temperatures', 'r')
        file1.close()



    except:
        ask = input("insert temperature points separated by space:").split()
        file1 = open('temperatures', 'w')
        for item in ask:
            file1.write("%s\n" % item)

        file1.close()

    file1 = open('temperatures', 'r')
    temp = file1.read()
    temp = temp.split("\n")
    temp.pop()
    file1.close()



    ask1 = input("Are we using FLS980 file? (y/n)")
    if ask1 == 'y':
        file2 = input("Enter file name:")
        #file2 = ("CHABRI-1_5p6K_to_300K.txt")
        with open(file2, 'r') as fin:
            data = fin.read().splitlines(True)
        with open('newfile.txt', 'w') as fout:
            fout.writelines(data[24:])

    else:
        print("Assuming .dat (Read Doc for help)")
        file2 = input("Enter file name:")
        with open(file2, 'r') as fin:
            data = fin.read().splitlines(True)
        with open('newfile.txt', 'w') as fout:
            fout.writelines(data[:])





    file3 = ("newfile.txt")

    T = pd.read_csv(file3, delimiter='\t', header=None)
    T.drop(T.columns[len(T.columns) - 1], axis=1, inplace=True)

    new_header = ['wavelengths'] + list(temp)
    T.columns = new_header
    T.to_csv('processed_data1.csv', sep='\t')
    # #
    coloumns = T.columns

    T.head()
    wave = T['wavelengths']
    wave = wave.to_numpy()

    temperatures = np.array(coloumns[1:], dtype=float)

    counts = T[T.columns[1:]].to_numpy()

    return wave, temperatures, counts
    
def postprocessing(temp,intensity, temp_inv, int_inv):
    processed_data = pd.DataFrame({'temperature': temp,

                                   'intensity': intensity,

                                   'temp_inv': temp_inv,

                                   'intensity_inv': int_inv})
    print(processed_data)
    ask = input("do you want to save the data? (y/n)")
    if ask == 'y':
        ask2 = input("Enter name of the file: ")
        processed_data.to_csv(str(ask2)+'.csv', index=False)
        print("Data saved!")

    return processed_data


def funca(x, a, b, c):
        return a * np.exp(-b * x) + c
def funcb(x, a1, b1, a2, b2, c):
        return a1 * np.exp(-b1 * x) + a2 * np.exp(-b2 * x) + c


#lmfit
def fitting_mono(temp_inv, int_inv, cropped_temp_inv,cropped_int_inv):

    if "popt" in locals():
        parameters = popt
    else:
        parameters = [0.003, 2325, 0.0000185]
    print("Fitting with monoexponential decay of the form y = (A*exp(-bx))+c")
    ask_guess = input("Do you wish to provide initial guess? (y/n)")
    if ask_guess == 'y':
        guess_be = input("type expected binding energy in meV: ")
        guess_be = float(guess_be)
        guess_para = guess_be * 100 * 0.116
        guess_ampli = input("type expected amplitude(A) value, we recommend " + str(parameters[0]) + " :")
        guess_ampli = float(guess_ampli)


    else:
        guess_para = parameters[1]
        guess_ampli = parameters[0]


    def resid(params, x=cropped_temp_inv, ydata=cropped_int_inv):

        a = params['a'].value
        b = params['b'].value
        c = params['c'].value

        y_model = funca(x, a, b, c)
        return y_model - ydata

    params = lmfit.Parameters()

    params.add('a', guess_ampli, min=0, max=1.0)
    params.add('b', guess_para, min=0, max=5000)
    params.add('c', 0.0000185, min=-1.0, max=1.0)

    o1 = lmfit.minimize(resid, params, args=(cropped_temp_inv, cropped_int_inv), method='leastsq')
    print("# Fit using leastsq:")
    lmfit.report_fit(o1)
    opt = []
    for name, param in o1.params.items():
        opt.append(param.value)
        print('{:7s} {:11.5f} {:11.5f}'.format(name, param.value, param.stderr))

    popt = np.array(opt)





    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.85)
    ax.set_title('Binding Energy fit', fontsize=14, fontweight='bold')
    f1 = ax.plot(temp_inv, int_inv, marker="^", markersize=4, color='grey')
    ax.plot(cropped_temp_inv, cropped_int_inv+o1.residual, color='crimson')
    ax.legend(['data', 'fitted region'], fontsize=12)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)

    ax.tick_params(direction='out', length=6, width=2, colors='k', labelsize=12)

    def format_func(value, tick_number):
        # find number of multiples of pi/2
        N = int(np.multiply(value, 10000000))

        return N

    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))

    ax.text(0.5, 0.5, r'Fit equation: $y= ae^{-bx}+c$', horizontalalignment='center',
     verticalalignment='center', transform=ax.transAxes, fontsize=12, fontweight='bold')
    ax.text(0.5, 0.45, "Parameters:", horizontalalignment='center',
     verticalalignment='center', transform=ax.transAxes, fontsize=12, fontweight='bold')
    ax.text(0.5,0.40, "a: " + str("{:.4f}".format(popt[0])), horizontalalignment='center',
     verticalalignment='center', transform=ax.transAxes, fontsize=12, fontweight='bold')
    ax.text(0.5, 0.35, "b: " + str("{:.2f}".format(popt[1])), horizontalalignment='center',
     verticalalignment='center', transform=ax.transAxes, fontsize=12, fontweight='bold')
    ax.text(0.5, 0.3, "c: " + str("{:.7f}".format(popt[2])), horizontalalignment='center',
     verticalalignment='center', transform=ax.transAxes, fontsize=12, fontweight='bold')
    ax.text(0.5, 0.25, "Binding energy: " + str(int((popt[1]) * 8.617 * 0.01)) + " meV", horizontalalignment='center',
     verticalalignment='center', transform=ax.transAxes, fontsize=12, fontweight='bold', color='firebrick')
    ax.set_xlabel('1/T (K)', fontweight='bold', fontsize=14)
    ax.set_ylabel('1/I (a.u.)', fontsize=14, fontweight='bold')
    plt.show()
    return popt


def fitting_bi(temp_inv, int_inv, cropped_temp_inv,cropped_int_inv):
    if "popt_bi" in locals():
        print('yes')
        parameters = popt_bi
    else:
        parameters = [5e-07, 142.86, 0.0105, 2700, -2e-07]
    print("Fitting with biexponential decay of the form y = (A1*exp(-b1x))+(A2*exp(-b2x))+c")
    ask_guess = input("Do you wish to provide initial guess? (y/n)")
    if ask_guess == 'y':
        guess_be_1 = input("type expected first binding energy (major contribution) in meV: ")
        guess_be_1 = float(guess_be_1)
        guess_para_1 = guess_be_1 * 100 * 0.116
        guess_be_2 = input("type expected second binding energy in meV: ")
        guess_be_2 = float(guess_be_2)
        guess_para_2 = guess_be_2 * 100 * 0.116
        guess_ampli_1 = input("type expected amplitude(A1) value, we recommend " + str(parameters[2]) + " :")
        guess_ampli_1 = float(guess_ampli_1)
        guess_ampli_2 = input("type expected amplitude(A2) value, we recommend " + str(parameters[0]) + " :")
        guess_ampli_2 = float(guess_ampli_2)


    else:
        guess_para_1 = parameters[1]
        guess_ampli_1 = parameters[0]
        guess_para_2 = parameters[3]
        guess_ampli_2 = parameters[2]




    def resid(params, x=cropped_temp_inv, ydata=cropped_int_inv):

        a1 = params['a1'].value
        b1 = params['b1'].value
        a2 = params['a2'].value
        b2 = params['b2'].value
        c = params['c'].value

        y_model = funcb(x, a1, b1, a2, b2, c)
        return y_model - ydata

    params = lmfit.Parameters()

    params.add('a1', guess_ampli_1, min=0, max=0.1)
    params.add('a2', guess_ampli_2, min=0, max=1.0)
    params.add('b1', guess_para_1, min=0, max=5000)
    params.add('b2', guess_para_2, min=0, max=5000)
    params.add('c', 0.0000185, min=-1.0, max=1.0)

    o2 = lmfit.minimize(resid, params, args=(cropped_temp_inv, cropped_int_inv), method='least_squares')
    lmfit.report_fit(o2)




    opt = []
    for name, param in o2.params.items():
        opt.append(param.value)
        print('{:7s} {:11.5f} {:11.5f}'.format(name, param.value, param.stderr))

    popt_bi = np.array(opt)
    print(popt_bi)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.85)
    ax.set_title('Binding Energy fit', fontsize=14, fontweight='bold')
    f1 = ax.plot(temp_inv, int_inv, marker="^", markersize=4, color='grey')
    ax.plot(cropped_temp_inv, cropped_int_inv + o2.residual,
            color='crimson')
    ax.legend(['data', 'fitted region'], fontsize=12)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)

    ax.tick_params(direction='out', length=6, width=2, colors='k', labelsize=12)

    def format_func(value, tick_number):
        # find number of multiples of pi/2
        N = int(np.multiply(value, 10000000))

        return N

    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))

    ax.text(0.6, 0.6, r'Fit equation: $y= a_{1}e^{-b_{1}x}+a_{2}e^{-b_{2}x}+c$', horizontalalignment='center',
     verticalalignment='center', transform=ax.transAxes, fontsize=12, fontweight='bold')
    ax.text(0.6, 0.55, "Parameters:", horizontalalignment='center',
     verticalalignment='center', transform=ax.transAxes, fontsize=12, fontweight='bold')
    ax.text(0.6, 0.5, "a1: " + str("{:.4f}".format(popt_bi[0])),horizontalalignment='center',
     verticalalignment='center', transform=ax.transAxes, fontsize=12, fontweight='bold')
    ax.text(0.6, 0.45, "b1: " + str("{:.4f}".format(popt_bi[1])),horizontalalignment='center',
     verticalalignment='center', transform=ax.transAxes, fontsize=12, fontweight='bold')
    ax.text(0.6, 0.40, "a2: " + str("{:.4f}".format(popt_bi[2])),horizontalalignment='center',
     verticalalignment='center', transform=ax.transAxes, fontsize=12, fontweight='bold')
    ax.text(0.6, 0.35, "b2: " + str("{:.4f}".format(popt_bi[3])),horizontalalignment='center',
     verticalalignment='center', transform=ax.transAxes, fontsize=12, fontweight='bold')
    ax.text(0.6, 0.3, "c: " + str("{:.7f}".format(popt_bi[4])),horizontalalignment='center',
     verticalalignment='center', transform=ax.transAxes, fontsize=12, fontweight='bold')
    ax.text(0.6, 0.25, "Binding energy 1: " + str("{:.3f}".format(float((popt_bi[1]) * 8.617 * 0.01))) + " meV", horizontalalignment='center',
     verticalalignment='center', transform=ax.transAxes,fontsize=14,
            fontweight='bold', color='firebrick')
    ax.text(0.6, 0.2, "Binding energy 2: " + str("{:.3f}".format(float((popt_bi[3]) * 8.617 * 0.01))) + " meV", horizontalalignment='center',
     verticalalignment='center', transform=ax.transAxes, fontsize=14,
            fontweight='bold', color='firebrick')
    ax.set_xlabel('1/T (K)', fontweight='bold', fontsize=14)
    ax.set_ylabel('1/I (a.u.)', fontsize=14, fontweight='bold')
    plt.show()

    return popt_bi

def plot_2D_pl(wave,temperatures,counts):
    plt.rcParams.update({'font.size': 16, 'font.weight': 'bold'})
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.85)
    ax.set_title('Emission scans', fontsize=14, fontweight='bold')
    for c in range(counts.shape[1]):
        ax.plot(wave, counts[:,c], label= str(temperatures[c]) + " K")

    if len(temperatures) < 30:
        ax.legend(fontsize=12)


    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)

    ax.tick_params(direction='out', length=6, width=2, colors='k', labelsize=12)

    def format_func(value, tick_number):

        N = int(np.divide(value, 1000))

        return N

    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))


    ax.set_xlabel('Wavelength (nm)', fontweight='bold', fontsize=14)
    ax.set_ylabel('Intensity (a.u.)', fontsize=14, fontweight='bold')
    xmin = min(wave)
    xmax = max(wave)
    ax.set_xlim([xmin, xmax])
    plt.show()
