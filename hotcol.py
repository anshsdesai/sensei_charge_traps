#!/usr/bin/env python
import sys, getopt
import array
import datetime
import numpy as np
import xml.etree.ElementTree as ET
from scipy.stats import poisson, binom, combine_pvalues
#from scipy.special import lambertw #for calculating mu from the 1e count
import struct
from collections import defaultdict
from numpy.lib.stride_tricks import as_strided

outfilename=""
findPixels = False
twoECols = True
inputXML = None
singlequad = None
nHDUs = None
rateMultiplier = 1.0
doChunkCut = True

options, remainder = getopt.gnu_getopt(sys.argv[1:], 'o:pti:q:N:R:Ch')

for opt, arg in options:
    if opt == '-o':
        outfilename = arg
    elif opt == '-p':
        findPixels = True
    elif opt == '-t':
        twoECols = False
    elif opt == '-i':
        inputXML = arg
    elif opt == '-q':
        singlequad = arg
    elif opt == '-N':
        nHDUs = int(arg)
    elif opt == '-R':
        rateMultiplier = float(arg)
    elif opt == '-C':
        doChunkCut = False
    elif opt == '-h':
        print("\nUsage: "+sys.argv[0]+" <output basename> <root files>")
        print("Arguments: ")
        print("\t-o: basename for output file")
        print("\t-p: find hot pixels")
        print("\t-t: don't check 2e for hot cols")
        print("\t-i: load input XML file and use as a starting point")
        print("\t-q: only process this quad (for testing)")
        print("\t-N: assume this number of HDUs for setting p-cuts")
        print("\t-R: a bigger number here increases the rate that is considered normal, and weakens the cut (default 1.0)")
        print("\t-C: don't check for chunks of hot columns")
        print("\n")
        sys.exit(0)

inhotcoldict = defaultdict(set)
if inputXML is not None:
    tree = ET.parse(inputXML)
    root = tree.getroot()
    for ccd in root.iter('ccdMask'):
        ccdnum = ccd.attrib['ltaname']
        for col in ccd.find('badCols'):
            hdunum = col.attrib['hdu']
            hduname = ccdnum+'_'+hdunum
            if col.tag=='cRange':
                inhotcoldict[hduname].update(range(int(col.attrib['x1']), int(col.attrib['x2'])+1))
            else:
                inhotcoldict[hduname].add(int(col.attrib['x']))

#import ROOT now, since otherwise PyROOT eats the command-line options
#from ROOT import gROOT, gStyle, TFile, TTree, TChain, TCanvas, gDirectory, TH1, TGraph, gPad, TF1, THStack, TLegend, TGraphErrors, TLatex, TEfficiency, TMath
from ROOT import gROOT
gROOT.SetBatch(True)
from ROOT import gStyle
from ROOT import TFile
from ROOT import TTree
from ROOT import TChain
from ROOT import TCanvas
from ROOT import gDirectory
from ROOT import TH1, TH1F
from ROOT import gPad
from ROOT import THStack
from ROOT import TLegend
from ROOT import TLatex
from ROOT import TEfficiency
from ROOT import TFeldmanCousins
from ROOT import TF1
from ROOT import SetOwnership


import skipper_utils


gStyle.SetOptStat(110011)
gStyle.SetOptFit(1)
#gStyle.SetPalette(57)

if (len(remainder)<1):
    print(sys.argv[0]+' <root files>')
    sys.exit()

infiles = remainder
#if len(infiles)>1:
#    infiles.sort(key=skipper_utils.decodeRunnum)

def stripFilename(name):
    name = name.split('/')[-1] #strip off directory path
    name = name.rpartition(".")[0] #strip off file extension, if any
    return name

#if (len(sys.argv)<3):
#    print("not enough input args")
#    sys.exit(1)
if outfilename=="":
    outfilename = stripFilename(infiles[0])
    if outfilename=="":
        outfilename = filename
    outfilename = "hotcol_"+outfilename
    print("no output filename supplied, using default: "+outfilename)


c = TCanvas("c","c",1200,900);
c.Print(outfilename+".pdf[")

outfile = TFile(outfilename+".root","RECREATE")

latex = TLatex()
latex.SetNDC(True)

histList = [] #not used, but needed so Python doesn't delete the histograms

gStyle.SetOptStat(0)

pol0 = TF1("pol0","[0]",0,10000)

np.set_printoptions(precision=2, linewidth=300)

#get list of HDUs
f = TFile(infiles[0]) #look in the first file
names = set([k.GetName() for k in f.GetListOfKeys()])
f.Close()
hdulist=[]
for name in names:#search for histos with names matching the pattern
    if name.startswith("hotpix_"):
        hdu = name[len("hotpix_"):]
        hdulist.append(hdu)
hdulist.sort()
if singlequad is not None:
    hdulist = [singlequad]
print("found HDUs:", hdulist)
HDU_LIST = hdulist
if nHDUs is None:
    nHDUs = len(HDU_LIST)

def combinePvals(pvalList): #a few ways to do this - we do the simplest for now
    return np.min(pvalList, axis=0)
    #return np.prod(pvalList,0)
    #return np.apply_along_axis(combine_pvalues, 0, pvalList)[1,...]

def windowSum(data, windowsize, ndim):
    """
    calculate the sliding-window sum of the N-dimensional data array
    """
    # based on https://stackoverflow.com/a/43087771
    window = (windowsize,)*ndim
    s = window + tuple(np.subtract(data.shape, window) + 1)
    return as_strided(data, shape=s, strides=data.strides*2).sum(axis=tuple(np.arange(ndim)))

def expandWindow(badX, windowsize):
    """
    take the coordinates of bad sliding windows, expand into the coordinates of all cells in those windows
    """
    ndim = len(badX)
    # badX is in np.nonzero() format of an ndim-tuple of 1-D arrays
    # convert to a single (N, ndim) array
    badX = np.transpose(badX)
    # create a (N, ndim) array of offsets
    offsets = np.transpose(np.ones((windowsize,)*ndim).nonzero())
    # add each offset to badX, concat into a single array
    expanded = np.concatenate([badX + offset for offset in offsets])
    # sort and deduplicate - np.unique doesn't seem to like empty input
    if len(expanded)>0:
        expanded = np.unique(expanded, axis=0)
    # convert back to the np.nonzero() format
    return tuple(expanded.T)

def rateModel(shape, multiplier=rateMultiplier):
    """
    return an array representing the worst-case event rate distribution across the cell array.
    there are three cases: uniform, linear X-dependence, linear Y-dependence
    linear X-dependence happens when spurious charge dominates
    linear Y-dependence happens when dark current dominates
    not clear this is the correct approach for 2+e events, but statistics are lower there so it doesn't really matter
    multiplier: scale factor to apply to the uniform model, for a weaker cut
    """
    nX = shape[0]
    uniform = np.full(shape, multiplier)
    linX = np.linspace(0.0, 2.0, nX)
    if len(shape)==1:
        return np.maximum(uniform,linX)
    else:
        nY = shape[1]
        linX = linX[:,np.newaxis]*np.ones((1,nY))
        linY = np.linspace(0.0, 2.0, nY)
        linY = linY[np.newaxis,:]*np.ones((nX,1))
        return np.maximum.reduce([uniform, linX, linY])

def calculateChunkPvals(goodCells, rateList, cellArrayList, nChunks):
    chunkPlist = np.zeros((len(cellArrayList),nChunks))
    #print(rateList[0])
    #if nDim==1:
    for i, cellArray in enumerate(cellArrayList):
        rates = rateModel(cellArray[:,0].shape)*rateList[i]
        chunks = zip(*[np.array_split(a,nChunks) for a in [cellArray, goodCells, rates]])
        #print([(cData[cGood,1].sum(), cData[cGood,0].sum()*2.0*rateList[i]) for cData, cGood in chunks])
        for iChunk, (cData, cGood, cRates) in enumerate(chunks):
            chunkPlist[i,iChunk] = poisson.sf(cData[cGood,1].sum()-0.5, (cData[cGood,0]*cRates[cGood]).sum())
    return chunkPlist

def calculatePvals(goodCells, cellArrayList, windowsize=1):
    # shape of the cell array
    cellShape = cellArrayList[0].shape[:-1]
    nDim = len(cellShape)

    # we will use this array to zero out the numerator and denominator counts of bad cells
    badCellsMask = np.ones(cellShape,dtype=bool)
    badCellsMask[goodCells] = 0

    # if windowsize>1, we will be shrinking the cell array
    cellShape = tuple(np.subtract(cellShape, windowsize) + 1)

    pvalList = np.zeros((len(cellArrayList),)+cellShape)
    rateList = np.zeros(len(cellArrayList))
    for i, cellArray in enumerate(cellArrayList):

        if windowsize==1:
            denomArray = cellArray[...,0]
            hitsArray = cellArray[...,1]
            goodDenom = denomArray[goodCells].sum()
            goodHits = hitsArray[goodCells].sum()
        else:
            # make copies of the arrays so we can modify them
            # this is necessary so we can calculate denom and hit counts for partially masked windows
            cellCopy = cellArray.copy()
            cellCopy[badCellsMask,:] = 0
            denomArray = cellCopy[...,0]
            hitsArray = cellCopy[...,1]
            goodDenom = denomArray.sum()
            goodHits = hitsArray.sum()
            if windowsize>1:
                denomArray = windowSum(denomArray, windowsize, nDim)
                hitsArray = windowSum(hitsArray, windowsize, nDim)

        if goodDenom==0:
            rateGood = 0.0
        else:
            rateGood = goodHits/goodDenom
        rateList[i] = rateGood

        #update p-values
        #survival function = probability of getting nEle (or greater) given mu=rate*nPix
        nExpected = denomArray*rateModel(denomArray.shape)*rateGood
        pvalList[i] = poisson.sf(hitsArray-0.5, nExpected)
    return rateList, pvalList

def addNeighbors(pvals, goodCells, badX, addCut):
    # whatever shape badX is, we want it to be a 1-D array
    badX = np.reshape(badX, -1)
    for iseed in badX:
        for iadd in range(iseed-1, -1, -1): #go backward
            if not goodCells[iadd] or pvals[iadd]>addCut: #already found/not a bad cell
                break
            #print("adding {0}, p={1}".format(iadd,pvals[iadd]))
            goodCells[iadd] = False
        for iadd in range(iseed+1, len(goodCells)): #go forward
            if not goodCells[iadd] or pvals[iadd]>addCut: #already found/not a bad cell
                break
            #print("adding {0}, p={1}".format(iadd,pvals[iadd]))
            goodCells[iadd] = False

def findBadCells(data, nCells, already_bad=None):
    """
    "cells" can be pixels, columns, whatever
    iterate:
    1. mark cells as bad "seeds" if they have an improbably high number of hits given the estimated rate
    2. add neighbors to the seeds if they also have elevated rates
    3. the average of the good cells is the new estimated rate
    we start by assuming all cells are good
    as we mark more cells as bad, the estimated rate goes down
    when the rate stops changing, we have a stable set of good cells
    nHDUs: if we're not analyzing all the HDUs, specify the number here to set p-cuts correctly
    """
    cellArrayList, nameList = zip(*data)
    n_runs = len(data)//2

    #expect to remove 0.5 cells over a quad for 1e, over all CCDs for 2+e
    pScales = [nCells, nCells*nHDUs]*n_runs # scale factor to convert local p-values to global
    pCut = 0.5
    chunkCut = 0.5
    addCut = 5e-2 #p-value threshold for also removing a neighbor
    nChunks = 16

    cellShape = cellArrayList[0].shape[:-1]
    goodCells = np.full(cellShape, True)
    if already_bad: # not None or empty
        goodCells[already_bad] = False

    nDim = len(cellShape)

    while True:
        while True: #run until we have a stable set of good cells
            oldGood = goodCells.copy()

            # start by looking at single cells; remove the hot ones
            # then look at 2-col or 2x2-pixel windows; remove the hot ones
            # etc. up to 5-col or 5x5
            for windowsize in range(1,6):
                # get average rates and pvals of windows
                rateList, pvalList = calculatePvals(goodCells, cellArrayList, windowsize)
                pvalList = [pval*pscale for pval,pscale in zip(pvalList, pScales)]
                pvals = combinePvals(pvalList)
                # get coords of bad windows
                badX = (pvals<pCut).nonzero()
                #print(np.sort(pvals[pvals>=pCut])[:10])
                # convert to coords of bad cells
                badX = expandWindow(badX, windowsize)
                #since the rate monotonically decreases, (pvals>=pCut) should also be a strictly shrinking set and we could just use that
                #but we have to do this so we don't lose any cells that were marked bad by previous steps (vhot, chunks)
                goodCells[badX] = False
                #print(windowsize, badX)

            # update with non-windowed, non-scaled pvals
            rateList, pvalList = calculatePvals(goodCells, cellArrayList)
            pvals = combinePvals(pvalList)
            if nDim==1: #check neighbors of bad cols
                # TODO: check neighbors of bad pix?
                # we want to find neighbors of all currently marked cols, not just the ones we found in this round, so we refresh badX from goodCells
                badX = np.logical_not(goodCells).nonzero()
                addNeighbors(pvals, goodCells, badX, addCut)

            nNew = np.count_nonzero(goodCells!=oldGood)
            if nNew==0: break
            print("removed {0} cells".format(nNew))
        
        if nDim==1 and doChunkCut:
            #print("look for chunks")
            chunkPlist = calculateChunkPvals(goodCells, rateList, cellArrayList, nChunks)
            chunkPlist = [pval*pscale for pval,pscale in zip(chunkPlist, pScales)]
            #print(chunkPlist)
            chunkPvals = combinePvals(chunkPlist)
            #print(chunkPvals)
            badChunks = (chunkPvals<chunkCut).nonzero()[0]
            if len(badChunks)==0: break #no bad chunks, we are done
            for iChunk in badChunks:
                np.copyto(np.array_split(goodCells,nChunks)[iChunk], False)
            print("removed {0} chunks".format(len(badChunks)))
        else:
            break

    # recalculate pvals with the final set of good cells
    rateList, pvalList = calculatePvals(goodCells, cellArrayList)

    badX = np.transpose(np.logical_not(goodCells).nonzero())
    #print(badX)
    #badX = badCells[:,2:].astype(int)
    if badX.shape[1]==1: #reduce to 1-D if we can
        badX = badX.ravel()
    else: #convert to list of tuples
        badX = list(map(tuple, badX))

    if nDim==1:
        cellType = "cols"
    else:
        cellType = "pix"
    print("found {0} bad {1}".format(len(badX), cellType))
    for i, name in enumerate(nameList):
        medianN = np.median(cellArrayList[i][goodCells,0])
        #cellArray, name in data:
        rateGood = rateList[i]
        print("{0}: rate {1}, median denominator {2}, expected count {3}, seed >{4}, add >{5}".format(name, rateGood, medianN, rateGood*medianN, poisson.isf(pCut/pScales[i], rateGood*medianN), poisson.isf(addCut/pScales[i], rateGood*medianN)))

    return (badX, goodCells, pvalList)

def plotPvals(goodCells, pvals, hname, stackGood, stackBad, minP, color):
    #make a histogram of the p-values, with logarithmic binning
    #add the histogram to the stack
    pedges = array.array('d')
    #minP = np.floor(np.log10(np.min(pvals)))
    #minP = -5.0
    maxP = 0.0
    nPbins = int(-10*minP)
    for ix in range(0,nPbins+1):
        pedges.append(10**(minP+(ix-0.5)*(maxP-minP)/(nPbins-1)))
    
    pvals_toplot = np.maximum(pvals,10**minP)
    hPvals_good = TH1F(hname+"_good", "p-values", nPbins, pedges)
    hPvals_good.SetLineColor(color)
    for p in pvals_toplot[goodCells]:
        hPvals_good.Fill(p)
    histList.append(hPvals_good)
    stackGood.Add(hPvals_good)
    
    hPvals_bad = TH1F(hname+"_bad", "p-values", nPbins, pedges)
    hPvals_bad.SetLineColor(color)
    for p in pvals_toplot[np.logical_not(goodCells)]:
        hPvals_bad.Fill(p)
    histList.append(hPvals_bad)
    stackBad.Add(hPvals_bad)

def readTH1D(h): #get the histogram's data array, discard over/underflow
    #note that TH2.Projection always returns a TH1D, even if you started with a TH2F
    #arrTemp = struct.unpack_from(str(h.GetSize())+'d', h.GetArray())
    arrTemp = np.frombuffer(h.GetArray(), dtype=np.float64, count=h.GetSize())
    return arrTemp[1:-1]
def readTH2F(h, nX, nY): #get the histogram's data array, reshape to 2-D and discard over/underflow
    #arrTemp = struct.unpack_from(str(h.GetSize())+'f', h.GetArray())
    arrTemp = np.frombuffer(h.GetArray(), dtype=np.float32, count=h.GetSize())
    return arrTemp.reshape(((nY+2),(nX+2)))[1:-1,1:-1].T
def readTH2D(h, nX, nY): #get the histogram's data array, reshape to 2-D and discard over/underflow
    #arrTemp = struct.unpack_from(str(h.GetSize())+'f', h.GetArray())
    arrTemp = np.frombuffer(h.GetArray(), dtype=np.float64, count=h.GetSize())
    return arrTemp.reshape(((nY+2),(nX+2)))[1:-1,1:-1].T

nplots = 3

coldatadict = {}
pixdatadict = {}
vhotcoldict = {} # cols that are so hot that they consistently have fewer unmasked pix - this cut runs before the hot col cut
hotcoldict = {}
vhotpixdict = {} # pix that are so hot that they are less often unmasked - this cut runs before the hot pix cut
hotpixdict = {}

for iFile, filename in enumerate(infiles):
    thefile = TFile(filename)
    outfile.cd()
    c.Divide(nplots,len(HDU_LIST))
    for iHdu,hdu in enumerate(HDU_LIST):
        print("reading {0}, HDU {1}".format(filename, hdu))
        if hdu not in hotcoldict: #initialize dictionaries
            coldatadict[hdu] = []
            pixdatadict[hdu] = []
            hotcoldict[hdu] = set()
            hotpixdict[hdu] = set()
            vhotcoldict[hdu] = set()
            vhotpixdict[hdu] = set()
        
        # apply the very-hot-column cut
        # we are looking for columns that have a lot of pixels that are always masked
        # this is a way to look for strong dark spikes or similar, which create a high-energy cluster in the same place in every image
        # these will evade the hot col cut because the hot col cut doesn't look at pixels near high-energy events

        # bleedless exposure per col
        bExpo = thefile.Get("hotcols_unmasked2d_{0}".format(hdu))
        nX = bExpo.GetXaxis().GetNbins()
        nY = bExpo.GetYaxis().GetNbins()
        minX = int(bExpo.GetXaxis().GetBinCenter(1)) #center of first bin
        dExpo = readTH2F(bExpo, nX, nY)
        pcut=.5/(nHDUs * nX)
        nim = np.sum(dExpo[0,:]) # count number of images by counting entries in an arbitrary column (this number is also used for the vhot pix cut later)
        pv = np.exp(np.log(pcut)/nim) # take the nth root of the desired pcut; this is the threshold that should be applied on the min of the pvalues
        coll = np.sum(dExpo, axis=0)
        # convert the pvalue threshold to a threshold on the number of unmasked pix
        fcut = np.nonzero(np.cumsum(coll/np.sum(coll)) > pv)[0][0]
        # find columns where no image has >=fcut unmasked pixels
        throw = np.nonzero(np.sum(dExpo[:,fcut:], axis=1) == 0)[0]
        print("Quadrant {0} very-hot col threshold: {1}".format(hdu, fcut))
        if len(throw) > 0:
            # convert from active area to image column numbers
            vhotcoldict[hdu].update(throw + minX)

        c.cd(nplots*iHdu+1)
        gPad.SetLogz(1)

        hotcols = thefile.Get("hotcols_{0}".format(hdu))
        histList.append(hotcols)
        hotcols.Draw("colz")

        c.cd(nplots*iHdu+2)
        gPad.SetLogz(1)

        hotcolscluster = thefile.Get("hotcolscluster_{0}".format(hdu))
        histList.append(hotcolscluster)
        hotcolscluster.Draw("colz")

        #number of unmasked pixels per col
        hotcolsDenom = hotcols.ProjectionX("hotcols_all_{0}_{1}".format(iFile,hdu))
        histList.append(hotcolsDenom)

        #technically, we should use binomial statistics for these? but mu is small so it doesn't matter
        #number of pixels with >=1e per col
        #hotcols1e = hotcols.ProjectionX("hotcols_1e_{0}_{1}".format(iFile,hdu),2,2)
        hotcols1e = hotcolscluster.ProjectionX("hotcols_1e_{0}_{1}".format(iFile,hdu),2,2)
        histList.append(hotcols1e)
        #number of pixels with >=2e per col
        if twoECols:
            #hotcols2e = hotcols.ProjectionX("hotcols_2e_{0}_{1}".format(iFile,hdu),3,-1)
            hotcols2e = hotcolscluster.ProjectionX("hotcols_2e_{0}_{1}".format(iFile,hdu),3,-1)
            histList.append(hotcols2e)

        c.cd(nplots*iHdu+3)
        gPad.SetLogy(1)
        hs = THStack("hs1d_{0}".format(hdu),"pixel counts")
        histList.append(hs)
        hs.Add(hotcolsDenom)
        #hs.Add(hotcolsN.Clone())
        hs.Add(hotcols1e)
        hs.Draw("nostack")

        nCols = hotcolsDenom.GetXaxis().GetNbins()
        minX = int(hotcolsDenom.GetXaxis().GetBinCenter(1)) #center of first bin
        maxX = int(hotcolsDenom.GetXaxis().GetBinCenter(nCols)) #center of last bin
        coldataDenom = readTH1D(hotcolsDenom)

        coldata1e = readTH1D(hotcols1e)
        col1eArray = np.zeros((maxX+1,2))
        col1eArray[minX:,0] = coldataDenom
        col1eArray[minX:,1] = coldata1e
        coldatadict[hdu].append((col1eArray, stripFilename(filename)+" cols (1 e)"))

        if twoECols:
            coldata2e = readTH1D(hotcols2e)
            col2eArray = np.zeros((maxX+1,2))
            col2eArray[minX:,0] = coldataDenom
            col2eArray[minX:,1] = coldata2e
            coldatadict[hdu].append((col2eArray, stripFilename(filename)+" cols (2+ e)"))

        if findPixels:
            hotpix = thefile.Get("hotpix_{0}".format(hdu))
            hotpixDenom = hotpix.Project3D("all_yx")
            hotpix.GetZaxis().SetRange(2,2)
            hotpix1 = hotpix.Project3D("1e_yx")
            hotpix.GetZaxis().SetRange(3,hotpix.GetZaxis().GetNbins()+1)
            hotpix2 = hotpix.Project3D("2e_yx")

            hotpixcluster = thefile.Get("hotpixcluster_{0}".format(hdu))
            hotpixcluster.GetZaxis().SetRange(2,2)
            hotpixcluster1 = hotpixcluster.Project3D("1e_yx")
            hotpixcluster.GetZaxis().SetRange(3,hotpix.GetZaxis().GetNbins()+1)
            hotpixcluster2 = hotpixcluster.Project3D("2e_yx")

            nPixX = hotpix1.GetXaxis().GetNbins()
            nPixY = hotpix1.GetYaxis().GetNbins()
            nPix = nPixX*nPixY
            pixdataDenom = readTH2D(hotpixDenom, nPixX, nPixY)
            pixdata1e = readTH2D(hotpixcluster1, nPixX, nPixY)
            pixdata2e = readTH2D(hotpixcluster2, nPixX, nPixY)
            minX = int(hotpix1.GetXaxis().GetBinCenter(1)) #center of first bin
            minY = int(hotpix1.GetYaxis().GetBinCenter(1)) #center of first bin
            maxX = int(hotpix1.GetXaxis().GetBinCenter(nPixX)) #center of last bin
            maxY = int(hotpix1.GetYaxis().GetBinCenter(nPixY)) #center of last bin

            pixArray1 = np.zeros((maxX+1, maxY+1, 2))
            pixArray2 = np.zeros((maxX+1, maxY+1, 2))
            pixArray1[minX:,minY:,0] = pixdataDenom
            pixArray1[minX:,minY:,1] = pixdata1e
            pixArray2[minX:,minY:,0] = pixdataDenom
            pixArray2[minX:,minY:,1] = pixdata2e
            
            pixdatadict[hdu].append((pixArray1, stripFilename(filename)+" pix (1 e)"))
            pixdatadict[hdu].append((pixArray2, stripFilename(filename)+" pix (2+ e)"))

            # apply the very-hot-pix cut
            meanDenom = pixdataDenom.mean() # mean number of times each pix is unmasked
            pDenom = meanDenom/nim # probability a pix is unmasked
            pcut = .5/(nHDUs * nPix)
            veryhotThresh = binom.ppf(q=pcut, n=nim, p=pDenom)
            print("Quadrant {0} very-hot pix threshold: {1} (out of {2} images)".format(hdu, veryhotThresh, nim))
            throw = (pixdataDenom < veryhotThresh).nonzero()
            throw = np.transpose((throw[0]+minX, throw[1]+minY))
            vhotpixdict[hdu].update(map(tuple, throw))

    c.cd()
    c.Print(outfilename+".pdf");
    c.Clear()

nplots = 4
c.Divide(nplots,len(HDU_LIST))
for iHdu,hdu in enumerate(HDU_LIST):
    inhot = inhotcoldict[hdu]
    vhot = vhotcoldict[hdu]
    allhot = vhot | inhot
    print("masking cols, HDU {0} ({1} input hot + {2} very hot = {3} already marked)".format(hdu, len(inhot), len(vhot), len(allhot)))

    badX, goodCols, pvalList = findBadCells(coldatadict[hdu], nCols, already_bad=(np.array(list(allhot),dtype=int),))
    hotcoldict[hdu].update(badX) #add these to the bad cols

    hsPcol1_good = THStack("hs_colpvals1e_{0}_good".format(hdu),"good col p-values, 1e")
    hsPcol2_good = THStack("hs_colpvals2e_{0}_good".format(hdu),"good col p-values, 2+ e")
    histList.append(hsPcol1_good)
    histList.append(hsPcol2_good)
    hsPcol1_bad = THStack("hs_colpvals1e_{0}_bad".format(hdu),"bad col p-values, 1e")
    hsPcol2_bad = THStack("hs_colpvals2e_{0}_bad".format(hdu),"bad col p-values, 2+ e")
    histList.append(hsPcol1_bad)
    histList.append(hsPcol2_bad)

    if twoECols:
        for iFile in range(len(pvalList)//2):
            plotPvals(goodCols, pvalList[2*iFile], "h_colpvals1e_{0}_{1}".format(iFile,hdu), hsPcol1_good, hsPcol1_bad, -5.0, iFile+1)
            plotPvals(goodCols, pvalList[2*iFile+1], "h_colpvals2e_{0}_{1}".format(iFile,hdu), hsPcol2_good, hsPcol2_bad, -5.0, iFile+1)
    else:
        for iFile in range(len(pvalList)):
            plotPvals(goodCols, pvalList[iFile], "h_colpvals1e_{0}_{1}".format(iFile,hdu), hsPcol1_good, hsPcol1_bad, -5.0, iFile+1)

    c.cd(nplots*iHdu+1)
    gPad.SetLogy(1)
    gPad.SetLogx(1)
    hsPcol1_good.Draw("nostack")
    c.cd(nplots*iHdu+2)
    gPad.SetLogy(1)
    gPad.SetLogx(1)
    hsPcol2_good.Draw("nostack")
    c.cd(nplots*iHdu+3)
    gPad.SetLogy(1)
    gPad.SetLogx(1)
    hsPcol1_bad.Draw("nostack")
    c.cd(nplots*iHdu+4)
    gPad.SetLogy(1)
    gPad.SetLogx(1)
    hsPcol2_bad.Draw("nostack")

c.cd()
c.Print(outfilename+".pdf");
c.Clear()

if findPixels:
    nplots = 4
    c.Divide(nplots,len(HDU_LIST))
    for iHdu,hdu in enumerate(HDU_LIST):
        vhot = vhotpixdict[hdu]
        print("masking pix, HDU {0}, {1} already marked as very hot".format(hdu, len(vhot)))

        #zero out the counts in bad cols - pretend they were already masked
        for cellArray, name in pixdatadict[hdu]:
            cellArray[list(hotcoldict[hdu]),:,:] = 0

        # rearrange the vhotpix set into a format we can use to index into an ndarray
        vhot = tuple(np.array(list(vhot)).T)

        badPix, goodPix, pvalList = findBadCells(pixdatadict[hdu], nPix, already_bad=vhot)
        hotpixdict[hdu].update(badPix)

        hsPpix1_good = THStack("hs_pixpvals1e_{0}_good".format(hdu),"good pix p-values, 1e")
        hsPpix2_good = THStack("hs_pixpvals2e_{0}_good".format(hdu),"good pix p-values, 2+ e")
        histList.append(hsPpix1_good)
        histList.append(hsPpix2_good)
        hsPpix1_bad = THStack("hs_pixpvals1e_{0}_bad".format(hdu),"bad pix p-values, 1e")
        hsPpix2_bad = THStack("hs_pixpvals2e_{0}_bad".format(hdu),"bad pix p-values, 2+ e")
        histList.append(hsPpix1_bad)
        histList.append(hsPpix2_bad)

        for iFile in range(len(pvalList)//2):
            plotPvals(goodPix, pvalList[2*iFile], "h_pixpvals1e_{0}_{1}".format(iFile,hdu), hsPpix1_good, hsPpix1_bad, -8.0, iFile+1)
            plotPvals(goodPix, pvalList[2*iFile+1], "h_pixpvals2e_{0}_{1}".format(iFile,hdu), hsPpix2_good, hsPpix2_bad, -8.0, iFile+1)
        #print(badPix)

        c.cd(nplots*iHdu+1)
        gPad.SetLogy(1)
        gPad.SetLogx(1)
        hsPpix1_good.Draw("nostack")
        c.cd(nplots*iHdu+2)
        gPad.SetLogy(1)
        gPad.SetLogx(1)
        hsPpix2_good.Draw("nostack")
        c.cd(nplots*iHdu+3)
        gPad.SetLogy(1)
        gPad.SetLogx(1)
        hsPpix1_bad.Draw("nostack")
        c.cd(nplots*iHdu+4)
        gPad.SetLogy(1)
        gPad.SetLogx(1)
        hsPpix2_bad.Draw("nostack")

        print("before merging: {0} hot pix, {1} hot cols".format(len(hotpixdict[hdu]), len(hotcoldict[hdu])))

        maxHotPix = max(10, 0.05*nPixY) # if we see this many hot pix in a col, we mark the col as hot
        maxHotPixRange = max(10, 0.1*nPixY) # if we see at least 3 hot pix, separated by this distance, we mark the col as hot
        pix2col = {}
        for x,y in hotpixdict[hdu]:
            if x not in pix2col:
                pix2col[x] = set()
            pix2col[x].add(y)
        for x,ySet in pix2col.items():
            yList = sorted(ySet)
            if ((len(yList)>2 and yList[-1]-yList[0]>maxHotPixRange) or len(yList)>maxHotPix):
                #print("found {0} pixels with x={1}: merging into a bad column".format(len(yList), x))
                hotcoldict[hdu].add(x)
                for y in yList:
                    hotpixdict[hdu].remove((x,y))
        print("after merging: {0} hot pix, {1} hot cols".format(len(hotpixdict[hdu]), len(hotcoldict[hdu])))

    c.cd()
    c.Print(outfilename+".pdf");
    c.Clear()

for hdu in sorted(hotcoldict):
    hotcolsmerged = []
    firstX = -99
    lastX = -99
    for x in sorted(hotcoldict[hdu]):
        if x==lastX+1:
            lastX = x
        else:
            if firstX>=0:
                if firstX==lastX:
                    hotcolsmerged.append(firstX)
                else:
                    hotcolsmerged.append((firstX, lastX))
            firstX = x
            lastX = x

    if firstX>=0:
        if firstX==lastX:
            hotcolsmerged.append(firstX)
        else:
            hotcolsmerged.append((firstX, lastX))
    #print("bad cols:", hotcolsmerged)
    if findPixels: print("HDU {0}: {1} bad cols, {2} bad pix".format(hdu, len(hotcoldict[hdu]), len(hotpixdict[hdu])))
    else: print("HDU {0}: {1} bad cols".format(hdu, len(hotcoldict[hdu])))
    hotcoldict[hdu] = hotcolsmerged
    #if findPixels: print("bad pix:", sorted(hotpixdict[hdu]))

if '_' in HDU_LIST[0]:
    ccdlist = list(set([hduname.split('_')[0] for hduname in HDU_LIST]))
    ccdlist.sort()
else:
    ccdlist = ['']

#print(ccdlist)
rootXML = ET.Element("ccdMasks")
for ccd in ccdlist:
    ccdXML = ET.Element('ccdMask')
    rootXML.append(ccdXML)
    ccdXML.set("ltaname",str(ccd))
    ccdXML.text='\n'
    ccdXML.tail='\n'
    hotcolXML = ET.SubElement(ccdXML,'badCols')
    hotcolXML.text='\n'
    hotcolXML.tail='\n'
    for hduname in sorted(hotcoldict):
        if ccd:
            thisccd,hdu = hduname.split('_')
            if thisccd!=ccd:
                continue
        else:
            hdu = hduname

        firstX = -99
        lastX = -99
        for x in hotcoldict[hduname]:
            if isinstance(x,tuple):
                colEle = ET.SubElement(hotcolXML,'cRange')
                colEle.set('hdu',str(hdu))
                colEle.set('x1',str(x[0]))
                colEle.set('x2',str(x[1]))
            else:
                colEle = ET.SubElement(hotcolXML,'column')
                colEle.set('hdu',str(hdu))
                colEle.set('x',str(x))
            colEle.tail='\n'

    # always make this element, even if we're not filling it - skExtract expects it
    hotpixXML = ET.SubElement(ccdXML,'badPixels')
    hotpixXML.text='\n'
    hotpixXML.tail='\n'
    if findPixels:
        for hduname in sorted(hotpixdict):
            if ccd:
                thisccd,hdu = hduname.split('_')
                if thisccd!=ccd:
                    continue
            else:
                hdu = hduname

            for x,y in sorted(hotpixdict[hduname]):
                pixEle = ET.SubElement(hotpixXML,'pixel')
                pixEle.set('hdu',str(hdu))
                pixEle.set('x',str(x))
                pixEle.set('y',str(y))
                pixEle.tail='\n'

    bleedcolXML = ET.SubElement(ccdXML,'bleedCols')
    bleedcolXML.text='\n'
    bleedcolXML.tail='\n'
    #bleededgeXML = ET.SubElement(ccdXML,'bleedXEdges')
    #bleededgeXML.text='\n'
    #bleededgeXML.tail='\n'

roottree = ET.ElementTree(rootXML)
roottree.write(outfilename+'.xml')


c.Print(outfilename+".pdf]");
outfile.Write()
outfile.Close()

sys.exit(0)


