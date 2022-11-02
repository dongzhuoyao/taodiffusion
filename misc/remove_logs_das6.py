"""
quota -sv

du -h --max-depth=1

"""
except_list = [
"logs/only image_2022-06-02T18-55-43_", #
"logs/share aece kl1e-7 normed_2022-06-03T16-43-17_",
"logs/celeba aece kl1e-7 lr1e-5_2022-06-03T16-48-47_", #AE for celeba
"logs/celeba share ae19h_2022-06-04T12-42-28_", #celeba share
"logs/perceiverio vocshare_2022-06-04T12-40-27_" #perceiverio vocshare
]

import os

command_list = []
for name in os.listdir('/home/thu/lab/vldm/logs'):
    if os.path.join("logs",name) in except_list:
        print('skip {}'.format(name))
    else:
        rm_str = "rm -rf '{}'".format(os.path.join("logs/",name))
        #print(rm_str)
        command_list.append(rm_str)

print(' && '.join(command_list))