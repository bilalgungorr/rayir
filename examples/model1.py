
import matplotlib.pyplot as plt
import openseespy.opensees as ops
import numpy as np
import rayir


def MC(b, h, d3, As1, As2, phi1, phiw, s, narm_xy, fc):

    n1  = int(round(4*As1/(np.pi*phi1**2)))
    n2  = int(round(4*As2/(np.pi*phi1**2)))

    rebar = [
            [n1, phi1],
            [n2, phi1] 
            ]

    #Ec = 3250*fc**0.5 + 14000 #  TS500-Denk.3.2
    Ec = 5000*fc**0.5 # TBDY
    A0 = (np.pi*phiw**2)/4
    Ast = [i*A0 for i in narm_xy]
    mesh_size = d3

    model = rayir.SectionAnalysis('aaa')

    model.make_geo()
    model.geo.set_mesh_info('tri', d3, mesher='builtin')
    model.geo.Rectangle(b, h, d3)
    model.geo.generate_mesh()

    model.make_rebar(phiw)
    model.rebar.positioning_rectangular_beam(rebar)


    model.make_material('mat_unconfined', 'mat_confined', 'mat_reinforcement' )
    model.mat_unconfined.Mander(fc*1.3, Ec, confined=False)
    model.mat_confined.Mander(fc*1.3, Ec, confined=True, s=s, Ast=Ast)
    model.mat_reinforcement.Steel(quality='S420', Es=2e5, fsy=420*1.3, fsu=550*1.3)

    es_cr, ec_cr = -1, -1
    model.assign_materials_to_parts(['mat_reinforcement', 'rebar', es_cr],
                                    ['mat_unconfined', 'unconfined'],
                                    ['mat_confined', 'confined', ec_cr])

    return model
    model.mat_reinforcement.Steel(quality='S420')

if __name__ == '__main__':
    #input: As1, As2,  phi1
    # b: genislik
    # h: yukseklik
    # d3: paspayi, etriyenin disindan itibaren
    # phiw: etriye capi
    # s: etriye adim mesafesi
    # arms: [x, y] etriye kolsayisi

    #### input #######
    b, h, d3 = 300, 500, 30
    phiw, s = 8, 100
    arms = [2, 2]

    As1 = 500
    As2 = 300
    phi1 = 14

    fc = 30
    P = b*h*fc*0.0


    curv_impact = np.linspace(0, 0.25, 150) 

    model = MC(b, h, d3, As1, As2, phi1, phiw, s, arms, fc)

    curvature, moment = model.MomentCurvature(curv_impact*1e-3, -P, direction=3)




    per =  rayir.Performance(model)
    per.perStrain('06Vy')

    #perRotation(Ls, Lp, phiy, phiu, My, eta=1):
    # Ls: kesme acikligi, moment/kesme kuvveti, kolon ve kirislerde L/2
    # eta: kiris ve kolonlarda 1, perdelerde 0.5
    Ls = 3000 # mm
    Lp = 200 # mm
    per.perRotation(Ls, Lp, eta=1)

    Ec = 5000*fc**0.5 # TBDY
    EIg = Ec*b*h**3/12
    phiy, My = per.pCM['main']['yield']

    print('='*70)
    print(f'EIe = {round(per.EIe/EIg, 4)} TBDY-Denk.5.2')
    print(f'EIe = {round(My/phiy/EIg, 4)} My/phiy')
    print('-'*20, 'rotation', '-'*20)
    
    print(f'{"level":15s} {"theta":10s}')
    for k, theta in per.tLimits.items():
        print(f'{k:10s} {theta:10.5f}')


    print('-'*20, 'strain', '-'*20)
    print(f'{"level":15s} {"phi":10s} {"mom":5s}')
    for k, [phi, mom] in per.pCM['main'].items():
        print(f'{k:10s} {phi*1e3:10.5f} {mom*1e-6:10.1f}')

    print('='*70)

    fig, ax = per.plot((10/2.54, 7/2.54), section=True)
    #fig.savefig('./sec1.pdf')
    plt.show()
