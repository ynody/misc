# vim: fileencoding=utf-8

import numpy
import math

class GFSKGen(object):
    """
    self.its -> 
    1e6/9600 mode s,
    40 gfsk
    """
    def __init__(self, its=8):
        self.its = its # sample rate[sps]/ symbol rate[sps] 1e6/9600. 
        self.makefilter()
    def makefilter(self):
        bt = 0.5 # BT積
        vlen = int(self.its)*1+2  # 片側の幅
        leng = int(self.its)* 3 # 最終的な系列長
        t = numpy.arange(leng)
        sig = math.sqrt( math.log(2.) ) / ( 2.* math.pi * bt )
        h = 1. / ( math.sqrt( 2.*math.pi) * sig * self.its ) * numpy.exp( -1*t*t/ (2.*sig*sig*self.its*self.its) )
        q = numpy.zeros(leng, dtype=float)
        # print( "len(q):", len(q) , " len(h):", len(h), "leng:", leng, " vlen:", vlen )
        q[:vlen-1] = h[1:vlen][::-1]
        q[vlen-1:2*vlen-1] = h[:vlen]
        k = q[:2*vlen-1] #numpy.concatenate( (h[1:40][::-1], h[:40]) )
        # print (" shape of k:", k.shape )
        asum = numpy.sum(k) # タップ係数の総和=1になるよう正規化．
        self.filtap = k / asum
    def get_filt(self):
        return self.filtap

def make_response( ovs=8, alpha=0.2, cut_sym=10 ):
    pi = numpy.pi
    t0 = ovs
    
    t = numpy.arange(ovs*cut_sym)

    special = t0 / (2*alpha)
    # print ( "s16pecial", special ) # 特異点
    if numpy.fabs(int(special) - special) < 1e-3:
        t[int(special)] = 0
        omegax = alpha * pi / t0
        spret = omegax/(2*pi) * numpy.sin(pi*pi/t0/(2*omegax))
    else:
        spret = None
        
    d2 = pi*pi/t0 * numpy.sinc(t/t0) * numpy.cos( alpha*  pi * t/t0 )/( pi*pi- 4*(alpha*pi*t/t0)**2 )
    if spret is not None:
        d2[int(special)] = spret

    return d2


def gen_nyqfil( ovs=8, alpha=0.5, cut_sym=10, sqr=True ):
    d = make_response( ovs, alpha, cut_sym )
    d2 = numpy.zeros(2*len(d)-1, dtype=complex)
    d2[len(d):] = d[1:]
    d2[:len(d)] = d[::-1]
    if sqr is True:
        d2 = numpy.roll( d2, len(d))
        fd2 = numpy.fft.fft(d2)
        fd2.imag = 0
        d2 = numpy.fft.ifft(numpy.sqrt(fd2))
        d2.imag = 0
        d2 = numpy.roll( d2, -len(d))
    return d2


import matplotlib.pyplot as plt

def test_gauss():
    a = GFSKGen(its=8)
    flt = a.get_filt()
    plt.plot( flt , "o-")

    d3 = numpy.fft.fft(flt) / numpy.sqrt(len(flt))
    d4 = ( d3*d3.conj() ).real
    # plt.plot( 20. * numpy.log10( numpy.sqrt(d4) ), "o-" )
    plt.show()

def test():
    plt.figure(0)
    d = make_response(8)

    d2 = numpy.zeros(2*len(d)-1, dtype=complex)
    d2[len(d):] = d[1:]
    d2[:len(d)] = d[::-1]
    plt.plot(d2)

    plt.figure(1)
    d3 = numpy.fft.fft(d2)
    d4 = (d3*d3.conj()).real
    plt.semilogy( d4 )

    plt.show()

def test2():
    d = gen_nyqfil(ovs=8, alpha=0.5, cut_sym=16, sqr=True)

    plt.figure(0)
    plt.plot(d.real)

    plt.figure(1)
    d3 = numpy.fft.fft(d)
    d4 = (d3*d3.conj()).real
    plt.semilogy( d4 )

    plt.show()


def test_conste():

    pat = numpy.random.randint( 0, 4, 1024 )
    tb = numpy.array( [ 1.+ 1.j, 1.-1.j, -1.-1.j, -1 + 1.j ] )
    txp = tb[pat]
    ovs = 8
    ztxp = numpy.zeros( len(txp)*ovs, dtype=complex)
    ztxp[::ovs] = txp
    filt_coef = gen_nyqfil(ovs=ovs, alpha=0.35, cut_sym=16, sqr=True)
    # print (filt_coef)
    zfiltered = numpy.convolve( ztxp, filt_coef, mode="valid")
    z2filtered = numpy.convolve( zfiltered, filt_coef, mode="valid")

    graph_d = z2filtered

    plt.figure(0, figsize=(6,6))
    plt.plot( graph_d.real, graph_d.imag, "-" )
    plt.figure(1)
    plt.plot( graph_d.real, "o-" )
    plt.plot( graph_d.imag, "o-" )
    plt.show()





if __name__ == "__main__":
    # test_gauss()
    test_conste()
    # test2()
