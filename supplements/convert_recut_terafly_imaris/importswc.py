# Export Filament
# By Sarun Gulyanon 28 July 2018
#
# <CustomTools>
#  <Menu>
#    <Item name="Import SWC as Filament" icon="Python">
#      <Command>PythonXT::ImportSWC(#i)</Command>
#    </Item>
#  </Menu>
# </CustomTools>

import ImarisLib
import Tkinter
import tkFileDialog
import numpy as np
import time

# Export Filament
# By Sarun Gulyanon 28 July 2018
#
# <CustomTools>
#  <Menu>
#    <Item name="Import SWC as Filament" icon="Python">
#      <Command>PythonXT::ImportSWC(#i)</Command>
#    </Item>
#  </Menu>
# </CustomTools>

import ImarisLib
import Tkinter
import tkFileDialog
import numpy as np
import time

def ImportSWC(aImarisId):
    try:
        # Create an ImarisLib object
        vImarisLib = ImarisLib.ImarisLib()
        # Get an imaris object with id aImarisId
        vImaris = vImarisLib.GetApplication(aImarisId)
        # Check if the object is valid
        if vImaris is None:
            print 'Could not connect to Imaris!'
            time.sleep(2)
            return
        
        vFactory = vImaris.GetFactory()
        vFilaments = vFactory.ToFilaments(vImaris.GetSurpassSelection())
        
        # get swc file to load
        root = Tkinter.Tk()
        root.withdraw()
        swcname = tkFileDialog.askopenfilename(title = "Select SWC file", filetypes = (("SWC files","*.swc"),("all files","*.*")))
        root.destroy()
        if not swcname: # asksaveasfilename return '' if dialog closed with "cancel".
            print 'No file was selected.'
            time.sleep(2)
            return
        print(swcname)
        swc = np.loadtxt(swcname)
    except Exception as e:
        print(e)
        time.sleep(20)
    
    # get pixel scale in XYZ resolution (pixel/um)
    V = vImaris.GetDataSet()
    pixel_scale = np.array([V.GetSizeX() / (V.GetExtendMaxX() - V.GetExtendMinX()), \
                            V.GetSizeY() / (V.GetExtendMaxY() - V.GetExtendMinY()), \
                            V.GetSizeZ() / (V.GetExtendMaxZ() - V.GetExtendMinZ())])
    pixel_offset = np.array([V.GetExtendMinX(), V.GetExtendMinY(), V.GetExtendMinZ()])

    print('pixel scale value: ', pixel_scale)
    print('minX, maxX: ', V.GetExtendMinX(), V.GetExtendMaxX())
    print('minY, maxY: ', V.GetExtendMinY(), V.GetExtendMaxY())
    print('minZ, maxZ: ', V.GetExtendMinZ(), V.GetExtendMaxZ())
    time.sleep(5)

    # ad-hoc fix X-flip when |maxX| < |minX|
    if abs(V.GetExtendMinX()) > abs(V.GetExtendMaxX()):
        print('Detected: |maxX| < |minX|')
        print('Perform X-flipping')
        time.sleep(3)
        # pixel_offset = np.array([V.GetExtendMinX(), V.GetExtendMinY(), V.GetExtendMaxZ()])
        pixel_offset[0] = V.GetExtendMaxX()
        pixel_scale[0] = -pixel_scale[0]

    # ad-hoc fix Y-flip when |maxY| < |minY|
    if abs(V.GetExtendMinY()) > abs(V.GetExtendMaxY()):
        print('Detected: |maxY| < |minY|')
        print('Perform Y-flipping')
        time.sleep(3)
        # pixel_offset = np.array([V.GetExtendMinX(), V.GetExtendMinY(), V.GetExtendMaxZ()])
        pixel_offset[1] = V.GetExtendMaxY()
        pixel_scale[1] = -pixel_scale[1]

    # ad-hoc fix Z-flip when |maxZ| < |minZ|
    if abs(V.GetExtendMinZ()) > abs(V.GetExtendMaxZ()):
        print('Detected: |maxZ| < |minZ|')
        print('Perform Z-flipping')
        time.sleep(3)
        # pixel_offset = np.array([V.GetExtendMinX(), V.GetExtendMinY(), V.GetExtendMaxZ()])
        pixel_offset[2] = V.GetExtendMaxZ()
        pixel_scale[2] = -pixel_scale[2]

    try: 
        # draw Filament
        N = swc.shape[0] # in this case, 51
        vFilaments = vImaris.GetFactory().CreateFilaments()
        vPositions = swc[:,2:5].astype(np.float) / pixel_scale
        # print('vPositions before offset: ', vPositions)
        # vPositions = swc[:,2:5].astype(np.flo at)
        vPositions = vPositions + pixel_offset
        # print('vPositions after offset: ', vPositions)
        
        ##### need correction if the y/z are flipped, otherwise comment these two line out ###### 
        # if abs(V.GetExtendMinY()) > abs(V.GetExtendMaxY()):
        #     vPositions[:,1] = V.GetExtendMaxY() +  (V.GetExtendMinY() - vPositions[:,1])
        # if abs(V.GetExtendMinZ()) > abs(V.GetExtendMaxZ()):  
        #     vPositions[:,2] = V.GetExtendMaxZ() +  (V.GetExtendMinZ() - vPositions[:,2])
        # print('vPositions after correction: ', vPositions)
        # time.sleep(10)

        vnumPoints = np.full(shape = N, fill_value = 1) # numer of points per filament
        vRadii     = swc[:,5].astype(np.float)
        vTypes     = np.zeros((N)) #(0: Dendrite; 1: Spine)
        vEdges     = np.empty((0,2))
        vnumEdges = np.full(shape = N, fill_value = 0)
        vTimeIndex = np.full(shape = N, fill_value = 0)
      
    except Exception as e:
        print(e)
        print('value creation failed')
        time.sleep(10)

    try: 
        vFilaments.AddFilamentsList(vPositions.tolist(), vnumPoints.tolist(), vRadii.tolist(), vTypes.tolist(), vEdges.tolist(), vnumEdges.tolist(),vTimeIndex.tolist())
        # vFilamentIndex = 0
        vVertexIndex = 1
    except Exception as e:
        print(e)
        print('create filament failed')
        time.sleep(10)

    # Add the filament object to the scene
    vScene = vImaris.GetSurpassScene()
    vScene.AddChild(vFilaments, -1)
    print('Import ' + swcname + ' completed')
    
    