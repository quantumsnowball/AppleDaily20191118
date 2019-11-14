# from ipdb import set_trace as br
# import sys, IPython; sys.excepthook = IPython.core.ultratb.ColorTB(call_pdb=True)
import os, argparse
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.optimize import minimize

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-s', '--start', type=str, default='20160101', help='The start date of all time series')
parser.add_argument('-e', '--end', type=str, default=None, help='The end date of all time series')
parser.add_argument('--minW', type=float, default=-0.20, help='min weight for each holdings')
parser.add_argument('--maxW', type=float, default=0.20, help='max weight for each holdings')
parser.add_argument('-mtr', '--minTargetR', type=float, default=+0.20, help='target return for max sharpe portfolio')
parser.add_argument('--maxAbsSumW', type=float, default=3.00, help='max absolute sum of all weights')
parser.add_argument('--minNetSumW', type=float, default=0.03, help='min net sum of all weights')
parser.add_argument('--maxNetSumW', type=float, default=1.00, help='max net sum of all weights')
parser.add_argument('-rf', '--riskfree', type=float, default=0.025, help='Risk-free rate')
sargs = parser.parse_args()

class Optimizer:
    def get(self, ticker, start, end):
        df = pd.read_csv(f'resources/{ticker}.csv', index_col=0, parse_dates=True)
        df = df.loc[start:end]
        return df
    
    def get_close(self, ticker, *args, **kwargs):
        return self.get(ticker, *args, **kwargs)['Close']

    def porf_mu(self, mus, weights):
        return np.dot(mus, weights)
    
    def porf_sigma(self, weights, covm):
        return np.sqrt(np.dot(np.dot(weights,covm), weights.T))

    def porf_sharpe(self, weights, mus, covm, verbose=False):
        pMu, pSigma = self.porf_mu(mus, weights), self.porf_sigma(weights, covm)
        sharpe = (pMu-self.riskfree)/pSigma
        return {'sharpe':sharpe, 'mu':pMu, 'sigma':pSigma} if verbose else sharpe

    def __init__(self, start=None, end=None, riskfree=None, minW=None, maxW=None):
        resources = [os.path.basename(x).replace('.csv','') for x in os.listdir('resources')]
        ref = self.get_close('0005.HK', start, end)
        tss = {t:self.get_close(t, start, end) for t in resources}
        tss = {t:ts for t,ts in tss.items() if len(ts)>0.95*len(ref)}
        tss = pd.concat(tss, axis=1)
        lnrs = np.log(tss).diff()
        mus = lnrs.mean()*252
        sigmas = lnrs.std()*np.sqrt(252)
        covm = lnrs.cov()*252
        
        self.mus, self.sigmas, self.covm = mus, sigmas, covm
        self.tickers = covm.columns.tolist()
        self.start, self.end = start, end
        self.minW, self.maxW = minW, maxW
        self.riskfree = riskfree

    def optimize(self, minTargetR=None, maxAbsSumW=None, minNetSumW=None, maxNetSumW=None):
        weights = np.random.random(len(self.tickers))-0.5
        
        # find max sharpe solution
        def min_expected_return(x, target=minTargetR):
            return self.porf_mu(self.mus, x)-target
        def min_weights_sums(x, target=minNetSumW):
            return np.sum(x)-target
        def max_weights_sums(x, target=maxNetSumW):
            return target-np.sum(x)
        def weights_abs_sums(x, target=maxAbsSumW):
            return target-np.sum(np.abs(x))
        print('Calculating maxSharpe portfolio ... ', end='')
        solution = minimize(lambda *args: -self.porf_sharpe(*args), weights, args=(self.mus, self.covm, ),
                            bounds=[(self.minW, self.maxW) for _ in range(len(self.tickers))],
                            constraints=(
                                {'type':'ineq', 'fun': weights_abs_sums},
                                {'type':'ineq', 'fun': min_expected_return}, 
                                {'type':'ineq', 'fun': min_weights_sums}, 
                                {'type':'ineq', 'fun': max_weights_sums}, 
                            ))

        print(f'result:{solution["success"]}')
        
        self.solution = {'solution':solution,
                        'metrics':self.porf_sharpe(solution['x'], self.mus, self.covm, verbose=True)}

    def plot(self):
        fig, ax = plt.subplots(1,1, figsize=(15,8))
        ax.axhline(0, c='k'); ax.axvline(0, c='k')
        ax.set_xlim(-0.05,self.sigmas.max())
        ax.set_ylim(self.mus.min(), self.mus.max())
        title = f'({self.start}-{self.end}), size:{len(self.tickers)}'
        ax.set_title(title)

        # plot individual stocks
        for ticker, mu, sigma in zip(self.tickers, self.mus, self.sigmas):
            ax.scatter(sigma, mu, c='gray')
            ax.annotate(ticker, (sigma, mu), c='gray')
        
        # print solution and metrics
        mt = self.solution['metrics']
        pf = pd.Series(self.solution['solution']['x'], index=self.tickers).sort_values(ascending=False)
        pf = pf[abs(pf)>0.00001]
        ax.scatter(mt['sigma'], mt['mu'], c='g', marker='X', s=200)
        text = f'P_max_sharpe\nmu:{mt["mu"]:.2%}, sd:{mt["sigma"]:.2%}\n#:{mt["sharpe"]:.2%}\n'
        text += f'abs.lv:{np.sum(np.abs(self.solution["solution"]["x"])):.2f}x\n'
        text += f'net.lv:{np.sum(self.solution["solution"]["x"]):.2f}x\n'
        text += '---------------------------\n'
        for t,w in (pf.head(10).append(pf.tail(10))).items():
            if abs(w)>0.001:
                text += f'{t} {w:8.2%}\n'
        ax.annotate(text, (mt['sigma'], mt['mu']), horizontalalignment='right', verticalalignment='top', c='b')

        plt.show(block=True)        

def main():
    app = Optimizer(
        start=sargs.start, end=sargs.end,
        riskfree=sargs.riskfree,
        minW=sargs.minW, maxW=sargs.maxW,
    )
    app.optimize(
        minTargetR=sargs.minTargetR, 
        maxAbsSumW=sargs.maxAbsSumW,
        minNetSumW=sargs.minNetSumW,
        maxNetSumW=sargs.maxNetSumW,
    )
    app.plot()

if __name__ == '__main__':
    main()