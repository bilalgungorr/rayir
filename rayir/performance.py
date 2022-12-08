from .bilinear import bilinear
import matplotlib.pyplot as plt
import numpy as np


def close_spines(ax, *where):
    for w in where:
        ax.spines[w].set_visible(False)


def findex(value, array):
    for i in range(len(array) - 1):
        if array[i] <= value <= array[i+1] or array[i] >= value >= array[i+1]:
            return i


def interpolate(x2, x0, x1, y0, y1):
    y2 = y1 + (x2 - x1)*(y1 - y0)/(x1 - x0)
    return y2


class Performance:

    def __init__(self, saModel):
        self.saModel = saModel
        mat_concrete = self.saModel.parts_mats_dict.get('confined')
        mat_steel = self.saModel.parts_mats_dict.get('rebar')

        if not mat_concrete:
            sys.exit('Must be used Confined Mander material model')

        if not mat_steel:
            sys.exit('Must be used Steel material model')


    def perRotation(self, Ls, Lp, eta=1):
        # db ortalama cap 
        # eta kolon-kirislerde 1, perdelerde 0.5

        mat_steel = self.saModel.parts_mats_dict.get('rebar')
        mat_concrete = self.saModel.parts_mats_dict.get('confined')

        dir_ind = abs(self.saModel.direction) - 2
        h = self.saModel.geo.height[dir_ind]
        
        diameters = []
        for value in self.saModel.rebar.rebar_dict.values():
            diameters.append(value[2])
        db = sum(diameters)/len(diameters)

        fc = mat_concrete._mat_function_args[0]
        fsy = mat_steel._mat_function_args[0]

        phiy, My = self.pCM['main']['yield']
        phiu = self.pCM['main']['ultimate'][0]


        # fsy: fye; fc : fce
        thetay = phiy*Ls/3 + 0.0015*eta*(1 + 1.5*h/Ls) + phiy*db*fsy/(8*fc**0.5)

        self.EIe = (My*Ls)/(thetay*3)

        tGO = (2/3)*((phiu - phiy)*Lp*(1 - 0.5*Lp/Ls) + 4.5*phiu*db)

        self.tLimits = {
            'yield': thetay,
            'MN': 0.0,
            'DC': 0.75*tGO,
            'CO': tGO
            }


    def perStrain(self, bilinear_method):

        curvature = self.saModel.curvature
        moment = self.saModel.moment
        
        mat_concrete = self.saModel.parts_mats_dict.get('confined')
        mat_steel = self.saModel.parts_mats_dict.get('rebar')
        
        ecGO = mat_concrete._ecGO
        eps_su = mat_steel.ultimate_strain
        esGO = 0.4*eps_su

        if ecGO > 0.018:
            ecGO = 0.018
        

        pCM = {'main': {}, 'ec': {}, 'es': {}}

        eLimits = {
                'ec': {
                    'MN': -0.0025,
                    'DC': -0.75*ecGO,
                    'CO': -ecGO,
                    'ultimate': mat_concrete.ultimate_strain
                    },
                'es': {
                    'MN': 0.0075,
                    'DC': 0.75*esGO,
                    'CO': esGO,
                    'ultimate': mat_steel.ultimate_strain
                    }
                }


        eStrains = {'ec': self.saModel.get_strain(part='confined', strain_sign=-1),
                    'es': self.saModel.get_strain(part='rebar', strain_sign=1)}


        CM_bilinear = bilinear(curvature, moment, method=bilinear_method).T.tolist()
        for j, level in enumerate(['yield']):#, 'ultimate']):
            pCM['ec'][level] = CM_bilinear[j+1]
            pCM['es'][level] = CM_bilinear[j+1]
            pCM['main'][level] = CM_bilinear[j+1]


        ind = findex(CM_bilinear[1][0], curvature)
        if ind:
            s1 = slice(ind, ind+2)
            eps_cy = interpolate(CM_bilinear[1][0], *curvature[s1], *eStrains['ec'][s1])
            eps_sy = interpolate(CM_bilinear[1][0], *curvature[s1], *eStrains['es'][s1])
            eLimits['ec']['yield'] = eps_cy
            eLimits['es']['yield'] = eps_sy



        for level in ('MN', 'DC', 'CO', 'ultimate'):
            for part in ('ec', 'es'):
                eps = eLimits[part][level]
                ind = findex(eps, eStrains[part])

                if ind and ind < len(curvature) - 2:
                    s1 = slice(ind, ind+2)
                else:
                    s1 = slice(-2, None)

                cur = interpolate(eps, *eStrains[part][s1], *curvature[s1])
                mom = interpolate(cur, *curvature[s1], *moment[s1])
                pCM[part][level] = [cur, mom]

            c_cm = pCM['ec'].get(level)
            s_cm = pCM['es'].get(level)

            if c_cm and s_cm:
                if s_cm[0] < c_cm[0]:
                    cm = s_cm
                else:
                    cm = c_cm

            elif c_cm:
                cm = c_cm
            elif s_cm:
                cm = s_cm

            pCM['main'][level] = cm

        self.eLimits = eLimits
        self.pCM = pCM 


    def plot(self, figsize, section=False):
        pCM = {}
        curvature = np.copy(self.saModel.curvature)*1e3
        moment = np.copy(self.saModel.moment)*1e-6


        allCM = []
        for k1, v1 in self.pCM.items():
            pCM[k1] = {}
            for k2, v2 in v1.items():
                pCM[k1][k2] = [v2[0]*1e3, v2[1]*1e-6]
                if k1 != 'main':
                    if not (k1 == 'ec' and k2 in ['yield', 'ultimate']):
                        allCM.append(pCM[k1][k2])

        allCM = np.array(allCM)


        xmax = max(curvature)*1.1


        plt.rcParams.update({
            'font.size': 7,
            'axes.titlesize': 8,
            'axes.labelsize': 7, 
            'figure.titlesize': 'small',
            'figure.frameon': False,
            'font.size': 7,
            'lines.linewidth': 1.,
            'xtick.labelsize': 7,
            'ytick.labelsize': 7, 
            'ytick.major.size': .0,
            'ytick.minor.size': .0,
            'xtick.major.size': .0,
            'xtick.minor.size': .0,
            'ytick.left': False,  
            'ytick.right': False,  
            'yaxis.labellocation': 'top',
            'xaxis.labellocation': 'right',
            'legend.fontsize': 7,
            'legend.labelspacing': 0.,
            'figure.autolayout': True,
            #'text.usetex': True,
            })

        ratios = [12, 1, 1]
        fig, ax = plt.subplots(3, 1,# sharex=True,
                #layout='tight',
                layout="constrained",
                figsize=figsize, dpi=100,
                gridspec_kw={'height_ratios': ratios}
                )
        #ax[0].autoscale_view('tight')


        ax[0].plot(curvature, moment, 'c-')

        #cm_y = pCM['bilinear']['yield']
        #cm_u = pCM['bilinear']['ultimate']
        cm_y = pCM['main']['yield']
        #cm_u = pCM['main']['ultimate']
        cm_u = curvature[-1], moment[-1]

        ax[0].plot([0, cm_y[0], cm_u[0]], [0, cm_y[1], cm_u[1]], 'm--', label='bilinear')

        close_spines(ax[0], 'top', 'right')
        close_spines(ax[1], 'top', 'right', 'left')
        close_spines(ax[2], 'top', 'right', 'left')

        markers = ['D', 'o', 's']
        markers = ['o', 'o', 'o']
        colors = {'yield': 'b', 'ultimate': 'k', 'MN': 'g', 'DC': 'orange', 'CO': 'r'}
        xlabels = [r'$\phi [1/m]$', r'$\epsilon_c [\%]$', r'$\epsilon_s [\%]$']


        ax[0].text(0., 1.01, 'M[kNm]',
                transform=ax[0].transAxes, ha='center', va='bottom')
        lw = plt.rcParams['lines.linewidth']
        for i, part in enumerate(['main', 'ec', 'es']):

            CM = pCM[part]
            xticks = []
            xtick_labels = []
            ax[i].text(1.01, 0, xlabels[i], 
                    transform=ax[i].transAxes, ha='left', va='center')
            for j, (level, cm) in enumerate(CM.items()):
                if i < 1:
                    ax[i].plot(*cm, c=colors[level], marker = markers[i], ms=3)
                    ax[i].axvline(cm[0], lw=0.7*lw, c='gray', alpha=0.5)

                else:

                    ax[i].plot(cm[0], 0, c=colors[level], marker=markers[i], ms=3)
                    xtick_labels.append(round(self.eLimits[part][level]*100, 2))

                xticks.append(cm[0])


            len_xticks = len(xticks)

            if i < 1:
                ax[i].set_ylim(bottom=0)
                ax[i].set_xticks(np.round(xticks, 3))
            else:
                for cm in allCM:
                    ax[i].axvline(cm[0], lw=0.7*lw, c='gray', alpha=0.5)
                ax[i].set_ylim(-0.5, 0.5)
                ax[i].set_yticks([])
                #xtick_labels = [round(i*100, 2) for i in list(self.eLimits[part].values())]
                ax[i].set_xticks(xticks, xtick_labels)
            ax[i].set_xlim(-0.01*xmax, xmax)


        pLevels = ('MN', 'DC', 'CO', 'yield', 'ultimate')
        for i, level in enumerate(pLevels):
            ax[0].plot([], [], lw=4, label=level,  c=colors[level])
        ax[0].legend(loc='lower right', bbox_to_anchor=(1., 0.0))

        if section:
            ax0 = fig.add_axes([0.35, 0.45, 0.35, 0.35], frameon=False)
            ax0.axis('equal')
            ax0.axis('off')
            self.saModel.rebar.plot(ax=ax0, show_id_phi=None)

        return fig, ax

