import os
def rename():
    path = './tree_rule_process/Cardiotocography/vrd_ddnnf_raw'  # 要统一修改的文件所在的文件夹路径
    filelist=os.listdir(path)
    for files in filelist:
        Olddir=os.path.join(path,files)#原来的文件路径
        if os.path.isdir(Olddir):  #如果当前是文件夹注意要跳过，主要是避免算法出错误
            continue
        filename=os.path.splitext(files)[0] #获取文件名  x.xx.cnf
        filetype=os.path.splitext(files)[1] #文件扩展名 .nnf
        newfile_name=filename.replace(".cnf","") #需要如何修改文件名
        Newdir = os.path.join(r'./tree_rule_process/Cardiotocography/vrd_ddnnf_raw', newfile_name + filetype)
        os.rename(Olddir,Newdir)

if __name__=='__main__':
    rename()