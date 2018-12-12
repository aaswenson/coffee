import pandas
import numpy as np
import matplotlib.pyplot as plt


def save_results(t, n, rho, p, Tf, c, dndt):
    """Save results of model to csv for future plotting.
    """
    names = ['times', 'npop', 'rho', 'Tf', 'c', 'power', 'dndt']
    savesteps = 10
    rows = len(t[0::savesteps])
    data = np.zeros(rows, dtype={'names' : names,
                                 'formats' : ['f8']*len(names)
                                })
    
    data['times'] = t[0::savesteps]
    data['npop'] = n[0::savesteps]
    data['rho'] = rho[0::savesteps]
    data['Tf'] = Tf[0::savesteps]
    data['power'] = p[0::savesteps]
    data['dndt'] = dndt[0::savesteps]
    data['c'] = [sum(x) for x in c][0::savesteps]


    np.savetxt('results.csv', data, delimiter=',', 
           fmt='%10.5f', header=','.join(names), comments='')

def load_from_csv(datafile="results.csv"):
    """load the results data from a csv.
    """
    data = pandas.read_csv(datafile)
    
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
                     'rho'   : 'reactivity [-]',
                     'Tf'    : 'Fuel Temperature [K]',
                     'c'     : 'Total Precursor Conc. [Bq]',
                     'power' : 'Thermal Power [W]',
                     'dndt'  : 'Neutron Pop. Time Rate of Change [n/s]'
                    }
    # plot
    fig = plt.figure()
    plt.plot(data[ind], data[dep])
    # titles and labels
    plt.title("{0} vs. {1}".format(dep, ind))
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
    plt.savefig(savename, dpi=1000, format='png')

    return plt


if __name__=='__main__':
    data = load_from_csv()
    plot_results(data, 'times', 'c', 'semilogy')
    plot_results(data, 'times', 'power', 'semilogy')
    plot_results(data, 'times', 'Tf')
    plot_results(data, 'times', 'rho')
    plot_results(data, 'times', 'npop', 'semilogy')
