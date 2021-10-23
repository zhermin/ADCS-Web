import time
import pyautogui as pag

w, h = pag.size()
wmid, hmid = w//2, h//2
EN = 2264
BS = 3491
EN_NEW = 2574
CP_19OCT = 256

missing = [3886,3967,4198,4210,4352,4435,4627,4646,4680,4710,4743,4797,4844,4904,4955] # (15)
# missing = [x-EN for x in missing]

num_imgs = CP_19OCT

region = (wmid-hmid, 0, h, h) # left, top, width, height

print('screenshotting...')

start = time.time()

imgs = []
for i in range(num_imgs):
  imgs.append(pag.screenshot(region=region))
  pag.click(wmid, hmid)
  time.sleep(0.3)

print('saving...')

for i in range(num_imgs):
  numbering = BS+EN+EN_NEW+1+i
  # numbering = missing[i]

  filename = f'data/klarf-19oct/wafer-{numbering:04}.jpg'
  imgs[i].save(filename)

print(f'{(time.time()-start)/60:.2f} mins')