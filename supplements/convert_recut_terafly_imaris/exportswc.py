# Export Filament
# By Sarun Gulyanon 28 July 2018
#
# <CustomTools>
#  <Menu>
#    <Item name="Export Filament as SWC" icon="Python">
#      <Command>PythonXT::ExportSWC(#i)</Command>
#    </Item>
#  </Menu>
# </CustomTools>

import ImarisLib
import Tkinter
import tkFileDialog
import numpy as np
import time

def ExportSWC(aImarisId):
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
    
    if vFilaments is None:
        print('Pick a filament first')
        time.sleep(2)
        return
    
    ###
    try:
        # get pixel scale in XYZ resolution (pixel/um)
        V = vImaris.GetDataSet()
        pixel_offset = np.array([V.GetExtendMinX(), V.GetExtendMinY(), V.GetExtendMinZ()])
        pixel_scale = np.array([V.GetSizeX() / (V.GetExtendMaxX() - V.GetExtendMinX()), \
                                V.GetSizeY() / (V.GetExtendMaxY() - V.GetExtendMinY()), \
                                V.GetSizeZ() / (V.GetExtendMaxZ() - V.GetExtendMinZ())])
        
        
        # ad-hoc fix X-flip when |maxX| < |minX|
        if abs(V.GetExtendMinX()) > abs(V.GetExtendMaxX()):
            pixel_offset[0] = V.GetExtendMaxX()
            pixel_scale[0] = -pixel_scale[0]

        # ad-hoc fix Y-flip when |maxY| < |minY|
        if abs(V.GetExtendMinY()) > abs(V.GetExtendMaxY()):
            pixel_offset[1] = V.GetExtendMaxY()
            pixel_scale[1] = -pixel_scale[1]

        # ad-hoc fix Z-flip when |maxZ| < |minZ|
        if abs(V.GetExtendMinZ()) > abs(V.GetExtendMaxZ()):
            pixel_offset[2] = V.GetExtendMaxZ()
            pixel_scale[2] = -pixel_scale[2]

    except Exception as e:
        print('error occurred during getting pixel scale in xyz resolution')
        print(e)
        time.sleep(20)

    try:
        # get filename
        root = Tkinter.Tk()
        root.withdraw()
        savename = tkFileDialog.asksaveasfilename(defaultextension=".swc")
        root.destroy()
        if not savename: # asksaveasfilename return '' if dialog closed with "cancel".
            print('No files selected')
            time.sleep(2)
            return
        print(savename)
    except Exception as e:
        print('error occured during creating file name of save file')
        print(e)
        time.sleep(20)

    
    try:
        # go through Filaments and convert to SWC format
        head = 0
        swcs = np.zeros((0,7)) 
        print('initial swcs is' + str(swcs)) # shape (0,7)
        
        vCount = vFilaments.GetNumberOfFilaments()
        print('vcount is ' + str(vCount)) # vcount=51

        for vFilamentIndex in range(vCount):
            vFilamentsXYZ = vFilaments.GetPositionsXYZ(vFilamentIndex)
            vFilamentsEdges = vFilaments.GetEdges(vFilamentIndex)
            vFilamentsRadius = vFilaments.GetRadii(vFilamentIndex)
            vFilamentsTypes = vFilaments.GetTypes(vFilamentIndex)
            vFilamentsTime = vFilaments.GetTimeIndex(vFilamentIndex)
          
            N = len(vFilamentsXYZ)

            # traverse through the Filament using BFS
            swc = np.zeros((1,7))  # refresh for each filament
            swc[0] = [head+1, vFilamentsTypes[0], 0, 0, 0, vFilamentsRadius[0], -1]

            pos = vFilamentsXYZ[0]
            # ###### 1. correction due to flipped Y and Z, use this code, comment 2 out ####
            # pos[0] = pos[0] - pixel_offset[0]
            # pos[1] = V.GetExtendMaxY() + V.GetExtendMinY() - pos[1] - pixel_offset[1]
            # pos[2] = V.GetExtendMaxZ() + V.GetExtendMinZ() - pos[2] - pixel_offset[2]

            ##### 2. no correction use this code, commen 1 out ####
            pos = vFilamentsXYZ[0] - pixel_offset


            swc[0, 2:5] = pos*pixel_scale

            # convert to integers since it's pixel values
            swc[0, 2:5] = swc[0, 2:5].round(0).astype(np.int)
            swcs = np.concatenate((swcs, swc), axis=0)
            head = head + 1

            # visited[0] = True
            # queue = [0]
            # prevs = [-1]
            # while queue: # only when queue has element
            #     cur = queue.pop() # cur=0 
            #     prev = prevs.pop()  # prev = -1
            #     swc[head] = [head+1, vFilamentsTypes[cur], 0, 0, 0, vFilamentsRadius[cur], prev]
            #     pos = vFilamentsXYZ[cur] - pixel_offset
            #     swc[head,2:5] = pos*pixel_scale # first row generated surccessfully

                # for idx in np.where(G[cur])[0]:
                #     if not visited[idx]:
                #         visited[idx] = True
                #         queue.append(idx)
                #         prevs.append(head+1)
                # head = head + 1

            # swcs = np.concatenate((swcs,swc),axis=0)
        # write to file
        np.savetxt(savename, swcs, '%d %d %d %d %d %d %d')
        print('Export to ' + savename + ' completed')
        time.sleep(3)

    except Exception as e:
        print('error occurred when converting filaments to swc files')
        print(e)
        time.sleep(20)

        
    
