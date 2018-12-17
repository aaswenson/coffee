import operator
import pandas
import numpy as np
import matplotlib.pyplot as plt


def save_results(data):
    """Save results of model to csv for future plotting.
    """
    
    names = list(data.dtype.names)
    precursors = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6']
    names += precursors
    formats = ['f8']*len(names)

    skip = 10
    savedata = np.zeros(len(data[0::skip]),
                        dtype={'names' : names,
                               'formats' : formats
                              })

    for col in data.dtype.names:
        if col == 'c':
            continue
        savedata[col][:] = data[col][0::skip]

    # save the precursor data individually
    for idx, c in enumerate(precursors):
        cs = [x[idx] for x in data['c'][0::skip]]
        savedata[c][:] = cs

    np.savetxt('results.csv', savedata, delimiter=',', 
           fmt='%10.5f', header=','.join(names), comments='')

def load_from_csv(datafile="results.csv"):
    """load the results data from a csv.
    """
    data = pandas.read_csv(datafile)
    
    return data

def unit_conv(conversions, data):
    """Apply unit conversions to the data.
    """
    opers = {'x' : np.multiply,
             '/' : np.divide,
             '+' : np.add
            }
    for conv in conversions:
        op = opers[conv.split()[1]]
        key = conv.split()[0]
        val = float(conv.split()[2])
        data[key] = op(data[key], val)

    return data

def filter_data(filters, data):

    """Apply useful filters on the data
    """
    opers = {'<' : operator.lt,
             '=' : operator.eq,
             '>' : operator.gt}
    
    for filter in filters:
        op= filter.split()[1]
        key = filter.split()[0]
        val = float(filter.split()[2])
        data = data[opers[op](data[key], val)]
    
    return data

def plot_results(data, ind, dep, log=None):
    """Generate Plots
    """
    label_strings = {'times' : 'Time [s]',
                     'npop'  : 'Neutron Population [-]',
                     'rho'   : 'reactivity [$]',
                     'Tf'    : 'Fuel Temperature [K]',
                     'c'     : 'Total Precursor Conc. [Bq]',
                     'power' : 'Thermal Power [W]',
                     'rho_insert' : 'Inserted Reactivity [$]',
                     'rho_feedback' : 'Reactivity Feedback [$]',
                     'dndt'  : 'Neutron Pop. Time Rate of Change [n/s]'
                    }
    titles = {'npop'  : {'times' : 'Neutron Population Through Startup'},
              'rho'   : {'times' : 'Reactivity'},
              'rho_feedback' : {'times' : 'Reactivity Temperature Feedback'},
              'rho_insert' : {'times' : 'Inserted Reactivity'},
              'c'     : {'times' : 'Precursor Concentration'},
              'power' : {'times' : 'Fission Power'},
              'Tf'    : {'times' : 'Fuel Temperature'}
             }
    # plot
    fig = plt.figure()
    plt.plot(data[ind], data[dep])
    # titles and labels
    plt.title(titles[dep][ind])
    plt.xlabel(label_strings[ind])
    plt.ylabel(label_strings[dep])

    if log == 'log-log':
        plt.xscale('log')
        plt.yscale('log')
    if log == 'semilogy':
        plt.yscale('log')
    if log == 'semilogx':
        plt.xscale('log')

    savename = '{0}_vs_{1}.png'.format(dep, ind)
    plt.savefig(savename, dpi=500, format='png')

    return plt


if __name__=='__main__':
    data = load_from_csv()
    data = unit_conv(['times / 60'], data)
#    data = filter_data(['times < 30'], data) 
    dol  = ['rho / 0.00642', 'rho_insert / 0.00642', 'rho_feedback / 0.00642']
    data = unit_conv(dol, data)
    plot_results(data, 'times', 'c', 'semilogy')
    plot_results(data, 'times', 'power')
    plot_results(data, 'times', 'Tf')
    plot_results(data, 'times', 'rho')
    plot_results(data, 'times', 'rho_insert')
    plot_results(data, 'times', 'rho_feedback')
    plot_results(data, 'times', 'npop')# 'semilogy')
