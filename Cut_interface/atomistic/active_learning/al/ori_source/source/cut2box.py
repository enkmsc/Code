import sys
import numpy as np
from ase.io import read, write
from ase.build.tools import cut

print("start_cut_box")

name = str(sys.argv[1])
cfg = read(name, format='lammps-data', style='atomic', Z_of_type={1: 3, 2: 15, 3: 16, 4: 35})
cellpar = cfg.get_cell()

box_length = 16
protect_length = 6.5
atom_num = 200
box_final = 16

while True:
    sto_mark = True
    box_length_scale = box_length/cellpar[0,0]
    cfg_cut = cut(cfg, a=(box_length_scale,0,0), b=(0,box_length_scale,0), c=(0,0,box_length_scale), origo=(-box_length_scale/2,-box_length_scale/2,-box_length_scale/2))
    cfg_cut = cfg_cut[np.argsort(cfg_cut.get_atomic_numbers())]
    
    #center atom
    cfg_cut.set_positions((cfg_cut.get_positions()-box_length/2))
    
    # put all atoms into box
    #pre_pos = cfg_cut.get_positions(wrap=False)
    #pre_pos[np.where(pre_pos > box_final/2-0.1)] = box_final/2-0.1
    #pre_pos[np.where(pre_pos < -box_final/2+0.1)] = -box_final/2+0.1
    #cfg_cut.set_positions(pre_pos)
    cfg_cut.set_cell(box_final * np.identity(3),scale_atoms=False)
    cfg_new = cfg_cut.copy()
    
    ato_num = cfg_cut.get_atomic_numbers()
    _, count = np.unique(ato_num, return_counts=True)
    
    if len(cfg_new) < 52:
        print("too less atoms, finish")
        break

    if np.shape(count)[0] == 4:
        print("initial atoms of the four elements: %s"%count)
        for i in range(1,100):
            count_del = count - i*np.array([6,1,5,1])
            if np.any(count_del < 0):
                count_del = count - (i-1)*np.array([6,1,5,1])
                print("atoms need to be deleted: %s"%count_del)
                break
            if 0 in count_del:
                print("atoms need to be deleted: %s"%count_del)
                break
        
        for ele_idx, ele_num in enumerate([3,15,16,35]):
            dis_vec = cfg_new.get_positions(wrap=False)
            
            idx_no_protect = []
            for i in range(np.shape(dis_vec)[0]):
                if np.any(np.abs(dis_vec[i]) > protect_length):
                    idx_no_protect.append(i)
            idx_no_protect = np.array(idx_no_protect)

            ato_num = cfg_new.get_atomic_numbers()
            idx_ele = np.intersect1d(np.where(ato_num == ele_num)[0], idx_no_protect)
            if idx_ele.size == 0:
                sto_mark = False
                print("WRONG! No boundary atoms can be removed")
            if sto_mark:
                ato_dis = cfg_new.get_all_distances(mic=True)
                print("min atomic distance: %s"%np.min(ato_dis[np.nonzero(ato_dis)]))
                min_all = np.zeros(np.shape(idx_ele))
                for i_idx, i in enumerate(idx_ele):
                    min_all[i_idx] = np.min(ato_dis[i,np.nonzero(ato_dis[i])])
                del_idx = idx_ele[np.argsort(min_all)[:count_del[ele_idx]]]
                del cfg_new[del_idx]

        ato_num = cfg_new.get_atomic_numbers()
        _, count = np.unique(ato_num, return_counts=True)
        print("end atoms of the four elements: %s"%count)
        ato_dis = cfg_new.get_all_distances(mic=True)
        print("final min atomic distance: %s"%np.min(ato_dis[np.nonzero(ato_dis)]))

        if len(cfg_new) > atom_num:
            if sto_mark == True:
                print("box length: %s"%box_length)
                test_dis = cfg_new.get_all_distances(mic=True)
                min_dis_cfg_test = np.min(test_dis[np.nonzero(test_dis)])
                print("min atomic distance after scaling: %s"%min_dis_cfg_test)
            
                cfg_new.set_positions((cfg_new.get_positions()+box_final/2))
                write("cutdata%s"%str(sys.argv[2]), cfg_new, format='lammps-data', specorder=["Li","P","S","Br"])
                break
            else:
                print("discard this data, stoichiometry incorrect")
                break
        else:
            print("too few atoms, increase box length")
            box_length += 0.2
    else:
        print("worng cfg, finish")
        break
