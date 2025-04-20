import win32com.client
import time
import datetime
import astropy.units as u
from astropy.coordinates import SkyCoord, AltAz
from astropy.time import Time


mount = win32com.client.Dispatch("ASCOM.CPWI.Telescope")
mount.Connected=True
print("设备已连接")
i=1
#银道坐标（l, b）
l = 0 
b = 0 
for l in range(355,330,-5):
    coord = SkyCoord(l=l*u.degree, b=b*u.degree, frame='galactic')
    ra_dec = coord.transform_to('icrs')
    ra = ra_dec.ra.degree
    dec = ra_dec.dec.degree
    print(datetime.datetime.now())
    print(f"l={l}°， RA = {ra}°, DEC = {dec}°")
    mount.SlewToCoordinates(ra/15, dec)
    while input()==1:
        break
mount.Connected=False





#mount.SlewToCoordinates(ra, dec)

#mount.SlewToCoordinates(0,0)  #RA,DEC
#mount.SlewToAltAz(0,0)
#mount.Connected=False
print("已经断开")
#time.sleep(10)


