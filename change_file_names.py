import os
os.chdir('C:\\Users\\Anonymous\\Documents\\Visual_Studio_files\\PyFiles\\Face_Recognition\\Data\\Anil')
i = 1
for file in os.listdir():
      src = file
      dst = "Anil"+str(i)+".jpg"
      os.rename(src, dst)
      i += 1
