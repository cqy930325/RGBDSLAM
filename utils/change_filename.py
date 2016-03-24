import os

rgb_path = "rgb/"
depth_path = "depth/"

with open("association_file.txt") as f:
    i = 1
    for line in f:
        names = line.split(' ')
        rgb = names[1]
        depth = names[3][:-1]
        os.rename(rgb, rgb_path+str(i)+'.png')
        os.rename(depth, depth_path+str(i)+'.png')
        i += 1
