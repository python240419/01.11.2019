#!/usr/bin/env python
# coding: utf-8

# In[4]:


def getParity(li):
    res = 0
    for e in li:
        res = res^e
    return res

print (getParity([1,0,1,0,1]))
print (getParity([True, False]))
print (getParity([True, True]))


# In[ ]:


## Card magic ##

import random

def getParityRow(code,r):
    parity = False
    for i in code[r]:
        parity ^= i
    return parity

def getParityCol(code,c):
    parity = False
    for i in range(len(code)):
        parity ^= code[i][c]
    return parity

def createRandCode(n):
    code = [[random.random() > 0.5 for i in range(n)] for j in range(n)]   
    for i in range(n):
        if getParityRow(code,i):
            code[i][-1] = not(code[i][-1])
            if getParityCol(code,i):
                code[-1][i] = not(code[-1][i])
    return code

def correctCode(code):
    r = -1
    c = -1
    n = len(code)
    for i in range(n):
        if getParityRow(code,i):
            r = i
            break
    for i in range(n):
        if getParityCol(code,i):
            c = i
            break
    if r != -1:
        code[r][c] = not(code[r][c])


N = 3
code = createRandCode(N)
print("code with n=3:", code)
print("")
i = random.randint(0,N-1)
j = random.randint(0,N-1)
code[i][j] = not(code[i][j])
print("flippd bit at [i,j]=",i,j)
print("code is now:", code)
print("")
correctCode(code)
print("After correction:", code)


# In[20]:


[print(r) for r in createRandCode(4)]


# In[16]:


## ID detection ##
def validControlDigit(id):
    nDigits = len(id)
    if nDigits != 9:    
        return False

    controlDigit = int(id[-1])
    idSum = 0
    for i in range(nDigits-1):
        curDigit = int(id[i])
        if i % 2 == 0:
            idSum += curDigit
        else:
            if curDigit < 5:
                idSum += curDigit * 2
            else:
                idSum += curDigit * 2 - 9

    return (idSum + controlDigit) % 10 == 0

ID1 = '032361271'
ID2 = '025096938'
print(validControlDigit(ID1))
print(validControlDigit(ID2))


# In[12]:


## RAID ##
from random import randint

def initDisks(nDrives=6, driveSize=5):
    ''' drives - number of disks with data (+1 for the RAID)
        driveSize - size of each drive
    '''
    nDrives = nDrives
    driveSize = driveSize
    data = [ [0]*driveSize for i in range(nDrives+1) ]
    return nDrives, driveSize, data

def randomizeDisks(data, nDrives,driveSize):
    for drive in range(nDrives):
        for position in range(driveSize):
            setData(data, nDrives, drive,position,randint(0,100))

def setData(data, nDrives, drive, position, value):
    '''set data in drive at position to value'''
    tmp = data[drive][position]
    data[drive][position] = value
    # update the control bit
    data[nDrives][position] = data[nDrives][position] ^ (tmp ^ data[drive][position])

def faultDisk(data, nDrives, faultDrive):
    '''drive turns to zeros'''
    data[faultDrive] = [0]*driveSize

def fixDisk(data, nDrives, driveSize, faultDrive):
    '''repair the data in drive'''
    for position in range(driveSize):
        for x in range(nDrives+1):
            if x != faultDrive:
                data[faultDrive][position] ^= data[x][position]                    

    
def showDisk(data):
    for drive in data:
        print(drive)
    print

nDrives, driveSize, data = initDisks()
randomizeDisks(data, nDrives,driveSize)
print ('original:')
showDisk(data)
faultDrive = 3
faultDisk(data, nDrives, faultDrive)
print ('fault:')
showDisk(data)
fixDisk(data, nDrives, driveSize, faultDrive)
print ('fixed:')
showDisk(data)


# In[ ]:


## Nearest Neighbor spell checker ##
def nearestNeighbor(word,words,distFunc):
    ''' word - word to be checked
        words - list of "correct" words
        compare - comparison measure between two words

        returns closest word'''
    bestMatch = ['',float('inf')]
    for curWord in words:
        score = distFunc(word,curWord)
        if score < bestMatch[1]:
            bestMatch[0] = curWord
            bestMatch[1] = score            
            if score == 0:
                return bestMatch
    return bestMatch

def spellingDict(correctWords,words,distFunc):
    spell = {}
    for word in words:
        nn = nearestNeighbor(word,correctWords,distFunc)        
        if nn[1] > 0: # no match
            #spell[word] = nn
            spell[word] = nn[0]
    return spell

def hammingDist(w1,w2):
    l1 = len(w1)
    l2 = len(w2)    
    score = abs(len(w1) - len(w2))
    for c1,c2 in zip(w1,w2):    
        if c1 != c2:
            score += 1
    return score

def printDict(d):
    for (key,val) in d.items():
        print(key, val)


def main(): 
    text = 'hwre ww sre in thr centrr pf tpwn'
    correctWords = ['the', 'in', 'of', 'center', 'we',
                    'here', 'are', 'town','yes']
    
    spell = spellingDict(correctWords,text.split(),hammingDist)
    printDict(spell)    
    

main()


# In[1]:


## Using genesis as dictionary and using it on Alice in Wonderland
"""
import urllib.request

def read_text_url(webpage):
    print(webpage)
    with urllib.request.urlopen(webpage) as url:
        text = str(url.read())
    return text


prefix = 'https://raw.githubusercontent.com/GITenberg/'
genesis_prefix = 'The-Bible-King-James-Version-Complete_30/master/'
alice_prefix = 'Alice-s-Adventures-in-Wonderland_11/master/'
    
genesis_text = read_text_url(prefix + genesis_prefix + '30.txt')
alice_text = read_text_url(prefix + alice_prefix + '11.txt')

spell = spellingDict(genesis_text.split(),alice_text.split(),hammingDist)
printDict(spell)
"""

# In[ ]:




