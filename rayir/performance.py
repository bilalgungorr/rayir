#!/usr/bin/python3
import matplotlib.pyplot as plt
from matplotlib.axis import Axis
# Algoritma
    # kritik strain degerlerini hesapla
    # kritik strain degerlerinin kaldigi araligin alt indisi bul
    # kritik straine tekabul eden egriligi bul
    # egrilige tekabul eden momenti bul


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

    def something(self, Ls, Lp, phiy, phiu, My, eta=1):
        # db ortalama cap 
        # eta kolon-kirislerde 1, perdelerde 0.5

        dir_ind = abs(self.saModel.direction) - 2
        h = self.saModel.geo.height[dir_ind]/1e3
        
        diameters = []
        for value in self.saModel.rebar.rebar_dict.values():
            diameters.append(value[2])
        db = sum(diameters)/len(diameters)/1e3

        for m in self.saModel.parts_mats_dict.values():
            if m.material_model == 'Confined Mander':
                fc = m._mat_function_args[0]
                fyh, eps_su = m._mat_function_args2[:2]
                fce = 1.3*fc
                fye = 1.2*fyh
               
        tpGO = (2/3)*((phiu - phiy)*Lp*(1 - 0.5*Lp/Ls) + 4.5*phiu*db)
        

        thetay = phiy*Ls/3 + 0.0015*eta*(1 + 1.5*h/Ls) + phiy*db*fye/(8*fce**0.5)
        print('thetay', thetay)
        EIe = (My*Ls)/(thetay*3)
        return EIe


    def performance(self, eps_c_phi_y, eps_s_phi_y):

        curvature, moment = self.saModel.curvature, self.saModel.moment
        
        mat_concrete = self.saModel.parts_mats_dict.get('confined')
        
        ecGO = mat_concrete.ecGO
        esGO = mat_concrete.esGO
        if ecGO > 0.018:
            ecGO = 0.018
        
        mat_steel = self.saModel.parts_mats_dict.get('rebar')
        eps_sy = mat_steel._mat_function_args[1]


        eLimits = {'ec': [-0.0025, -abs(eps_c_phi_y), -0.75*ecGO, -ecGO],
                   'es': [0.0075, eps_s_phi_y, 0.75*esGO, esGO]} 

        eStrains = {'ec': self.saModel.get_strain(coordinates='unconfined'),
                    'es': self.saModel.get_strain(rebar_tag=-1)}
        

        sCM = {'ec': [[], []],
               'es': [[], []]}

        for part in sCM:
            for e in eLimits[part]:
                # TODO: condition, index not in interval. 
                ind = findex(e, eStrains[part])
                if ind:
                    s1 = slice(ind, ind+2)
                    s2 = slice(ind+1, ind+3)
                    cur = interpolate(e, *eStrains[part][s1], *curvature[s2])
                    mom = interpolate(cur, *curvature[s2], *moment[s2])
                    
                    sCM[part][0].append(cur)
                    sCM[part][1].append(mom)

        
        
        ultCM = [curvature[-1], moment[-1]]

        lenec, lenes = [len(sCM[i][0]) for i in ['ec', 'es']]
        maxind = max(lenec, lenes)
        mainsCM = [[0]*maxind, [0]*maxind]

        for i in range(maxind):
            if i < lenec:
                ecC, ecM = sCM['ec'][0][i], sCM['ec'][1][i]
                
                if i < lenes:
                    esC, esM = sCM['es'][0][i], sCM['es'][1][i]
                    cond = 1*(ecC > esC) 
                    mainsCM[0][i] = [ecC, esC][cond]
                    mainsCM[1][i] = [ecM, esM][cond]

                else:
                    mainsCM[0][i] = ecC
                    mainsCM[1][i] = ecM

            else:
                mainsCM[0][i] = sCM['es'][0][i]
                mainsCM[1][i] = sCM['es'][1][i]

        self.eLimits = eLimits
        self.sCM = sCM 
        self.mainsCM = mainsCM
        self.labels = [*['MN', r'$\phi_y$', 'DC', 'CO'][:maxind], r'$\phi_u$']
        self.xticks = [*mainsCM[0], ultCM[0]]
        self.yticks = [*mainsCM[1], ultCM[1]]

        # sil
        self.eStrains = eStrains


    def plot(self, fig, precision=4, section=False, rotation_xaxis=0):
        sCM = self.sCM 
        eLimits = {part: [round(i, precision) for i in values] 
                    for part, values in self.eLimits.items()}
        curvature, moment = self.saModel.curvature, self.saModel.moment
        xticks = [round(i, precision) for i in self.xticks] 
        yticks = [int(i) for i in self.yticks] 
        labels =  self.labels

        xmax = max(curvature)*1.05

        if fig == -1:
            fig = plt.figure(figsize=(9, 6), dpi=80)

        fgsize = 0.05
        dy = fgsize*1.6
        ax = fig.add_axes([0, 2*(fgsize + dy) + 0.5*fgsize, 
                           1, 1 - 2*(fgsize + dy) - 0.5*fgsize], frameon=False)
        ax.set_xlim(0, xmax)
        ax.set_xticks(xticks)
        plt.xticks(rotation=rotation_xaxis)
        ax.set_ylim(0, max(moment)*1.1)
        ax.grid(axis='x')
        ax.set_xlabel(r'$\phi (m^{-1})$')
        ax.xaxis.set_label_coords(1.05, 0, transform=None)
        ax.axhline(0, lw=3)
        ax.axvline(0, lw=3)
        
        axt = ax.twiny()
        axt.set_frame_on(False)   
        axt.set_xlim(0, xmax)
        axt.set_xticks(xticks)
        axt.set_xticklabels(yticks)
        axt.set_xlabel(r'$M (kNm)$')
        axt.xaxis.set_label_coords(0, 1.05, transform=None)

        ax.plot(curvature, moment, '.--', ms=2)

        for j in range(len(labels)):
            ax.text(xticks[j], yticks[j], labels[j])


        axs = [fig.add_axes([0, 0, 1, fgsize], frameon=False),
               fig.add_axes([0, fgsize + dy, 1, fgsize], frameon=False)]

        xnames = [r'$\epsilon_c$', r'$\epsilon_s$']
        for j, [key, CM] in enumerate(sCM.items()):

            axs[j].yaxis.set_visible(False)
            axs[j].xaxis.set_label_coords(1.05, 0, transform=None)
            axs[j].axhline(0, lw=1.5)
            axs[j].grid(axis='x')
            
            axs[j].set_xlim(0, xmax)
            axs[j].set_ylim(-0.5, 1)
            axs[j].set_xticks(CM[0])
            axs[j].set_xticklabels([abs(val) for val in eLimits[key][:len(CM[0])]])
            axs[j].set_xlabel(xnames[j])
            
            for i in range(len(CM[0])):
                axs[j].text(CM[0][i], 0.2, labels[i])

        if section:
            ax0 = fig.add_axes([0.5, 0.3, 0.4, 0.4], frameon=False)
            ax0.axis('equal')
            ax0.axis('off')
            self.saModel.rebar.plot(ax=ax0, show_id_phi='phi')

#
