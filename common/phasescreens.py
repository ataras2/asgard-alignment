"""
forked from aotools github https://github.com/AOtools/aotools/blob/main/aotools/turbulence/infinitephasescreen.py. 
Removed numba dependency due to numpy conflicts for stability.


"""


"""
Infinite Phase Screens
----------------------

An implementation of the "infinite phase screen", as deduced by Francois Assemat and Richard W. Wilson, 2006.
"""

from scipy import linalg
import numpy
from numpy import fft
from scipy.special import gamma, kv
#import numba



def ft_sh_phase_screen(r0, N, delta, L0, l0, FFT=None, seed=None):

    """
    Creates a random phase screen with Von Karmen statistics with added
    sub-harmonics to augment tip-tilt modes.
    (Schmidt 2010)

    .. note::
        The phase screen is returned as a 2d array, with each element representing the phase 
        change in **radians**. This means that to obtain the physical phase distortion in nanometres, 
        it must be multiplied by (wavelength / (2*pi)), (where `wavellength` here is the same wavelength
        in which r0 is given in the function arguments)

    Args:
        r0 (float): r0 parameter of scrn in metres
        N (int): Size of phase scrn in pxls
        delta (float): size in Metres of each pxl
        L0 (float): Size of outer-scale in metres
        l0 (float): inner scale in metres
        seed (int, optional): seed for random number generator. If provided, 
            allows for deterministic screens  

    Returns:
        ndarray: numpy array representing phase screen in radians
    """
    R = numpy.random.default_rng(seed)

    D = N * delta
    # high-frequency screen from FFT method
    phs_hi = ft_phase_screen(r0, N, delta, L0, l0, FFT, seed=seed)

    # spatial grid [m]
    coords = numpy.arange(-N/2,N/2)*delta
    x, y = numpy.meshgrid(coords,coords)

    # initialize low-freq screen
    phs_lo = numpy.zeros(phs_hi.shape)

    # loop over frequency grids with spacing 1/(3^p*L)
    for p in range(1,4):
        # setup the PSD
        del_f = 1 / (3**p*D) #frequency grid spacing [1/m]
        fx = numpy.arange(-1,2) * del_f

        # frequency grid [1/m]
        fx, fy = numpy.meshgrid(fx,fx)
        f = numpy.sqrt(fx**2 +  fy**2) # polar grid

        fm = 5.92/l0/(2*numpy.pi) # inner scale frequency [1/m]
        f0 = 1./L0

        # outer scale frequency [1/m]
        # modified von Karman atmospheric phase PSD
        PSD_phi = (0.023*r0**(-5./3)
                    * numpy.exp(-1*(f/fm)**2) / ((f**2 + f0**2)**(11./6)) )
        PSD_phi[1,1] = 0

        # random draws of Fourier coefficients
        cn = ( (R.normal(size=(3,3))
            + 1j*R.normal(size=(3,3)) )
                        * numpy.sqrt(PSD_phi)*del_f )
        SH = numpy.zeros((N,N),dtype="complex")
        # loop over frequencies on this grid
        for i in range(0, 3):
            for j in range(0, 3):

                SH += cn[i,j] * numpy.exp(1j*2*numpy.pi*(fx[i,j]*x+fy[i,j]*y))

        phs_lo = phs_lo + SH
        # accumulate subharmonics

    phs_lo = phs_lo.real - phs_lo.real.mean()

    phs = phs_lo+phs_hi

    return phs


def ft_phase_screen(r0, N, delta, L0, l0, FFT=None, seed=None):
    """
    Creates a random phase screen with Von Karmen statistics.
    (Schmidt 2010)
    
    Parameters:
        r0 (float): r0 parameter of scrn in metres
        N (int): Size of phase scrn in pxls
        delta (float): size in Metres of each pxl
        L0 (float): Size of outer-scale in metres
        l0 (float): inner scale in metres
        seed (int, optional): seed for random number generator. If provided, 
            allows for deterministic screens  

    .. note::
        The phase screen is returned as a 2d array, with each element representing the phase 
        change in **radians**. This means that to obtain the physical phase distortion in nanometres, 
        it must be multiplied by (wavelength / (2*pi)), (where `wavellength` here is the same wavelength
        in which r0 is given in the function arguments)

    Returns:
        ndarray: numpy array representing phase screen in radians
    """
    delta = float(delta)
    r0 = float(r0)
    L0 = float(L0)
    l0 = float(l0)

    R = numpy.random.default_rng(seed)

    del_f = 1./(N*delta)

    fx = numpy.arange(-N/2., N/2.) * del_f

    (fx, fy) = numpy.meshgrid(fx,fx)
    f = numpy.sqrt(fx**2. + fy**2.)

    fm = 5.92/l0/(2*numpy.pi)
    f0 = 1./L0

    PSD_phi = (0.023*r0**(-5./3.) * numpy.exp(-1*((f/fm)**2)) / (((f**2) + (f0**2))**(11./6)))

    PSD_phi[int(N/2), int(N/2)] = 0

    cn = ((R.normal(size=(N, N))+1j * R.normal(size=(N, N))) * numpy.sqrt(PSD_phi)*del_f)

    phs = ift2(cn, 1, FFT).real

    return phs


def ift2(G, delta_f, FFT=None):
    """
    Wrapper for inverse fourier transform

    Parameters:
        G: data to transform
        delta_f: pixel seperation
        FFT (FFT object, optional): An accelerated FFT object
    """

    N = G.shape[0]

    if FFT:
        g = numpy.fft.fftshift(FFT(numpy.fft.fftshift(G))) * (N * delta_f) ** 2
    else:
        g = fft.ifftshift(fft.ifft2(fft.fftshift(G))) * (N * delta_f) ** 2

    return g

def phase_covariance(r, r0, L0):
    """
    Calculate the phase covariance between two points seperated by `r`, 
    in turbulence with a given `r0 and `L0`.
    Uses equation 5 from Assemat and Wilson, 2006.

    Parameters:
        r (float, ndarray): Seperation between points in metres (can be ndarray)
        r0 (float): Fried parameter of turbulence in metres
        L0 (float): Outer scale of turbulence in metres
    """
    # Make sure everything is a float to avoid nasty surprises in division!
    r = numpy.float32(r)
    r0 = float(r0)
    L0 = float(L0)

    # Get rid of any zeros
    r += 1e-40

    A = (L0 / r0) ** (5. / 3)

    B1 = (2 ** (-5. / 6)) * gamma(11. / 6) / (numpy.pi ** (8. / 3))
    B2 = ((24. / 5) * gamma(6. / 5)) ** (5. / 6)

    C = (((2 * numpy.pi * r) / L0) ** (5. / 6)) * kv(5. / 6, (2 * numpy.pi * r) / L0)

    cov = A * B1 * B2 * C

    return cov


class PhaseScreen(object):
    """
    A "Phase Screen" for use in AO simulation.  Can be extruded infinitely.

    This represents the phase addition light experiences when passing through atmospheric 
    turbulence. Unlike other phase screen generation techniques that translate a large static 
    screen, this method keeps a small section of phase, and extends it as necessary for as many
    steps as required. This can significantly reduce memory consumption at the expense of more
    processing power required.

    The technique is described in a paper by Assemat and Wilson, 2006 and expanded upon by Fried, 2008.
    It essentially assumes that there are two matrices, "A" and "B",
    that can be used to extend an existing phase screen.
    A single row or column of new phase can be represented by 

        X = A.Z + B.b

    where X is the new phase vector, Z is some data from the existing screen,
    and b is a random vector with gaussian statistics.

    This object calculates the A and B matrices using an expression of the phase covariance when it
    is initialised. Calculating A is straightforward through the relationship:

        A =  Cov_xz . (Cov_zz)^(-1).

    B is less trivial.

        BB^t = Cov_xx - A.Cov_zx

    (where B^t is the transpose of B) is a symmetric matrix, hence B can be expressed as 

        B = UL, 

    where U and L are obtained from the svd for BB^t

        U, w, U^t = svd(BB^t)

    L is a diagonal matrix where the diagonal elements are w^(1/2).    

    On initialisation an initial phase screen is calculated using an FFT based method.
    When 'add_row' is called, a new vector of phase is added to the phase screen.

    Existing points to use are defined by a "stencil", than is set to 0 for points not to use
    and 1 for points to use. This makes this a generalised base class that can be used by 
    other infinite phase screen creation schemes, such as for Von Karmon turbulence or 
    Kolmogorov turbulence.

    .. note::
        The phase screen is returned on each iteration as a 2d array, with each element representing the phase 
        change in **radians**. This means that to obtain the physical phase distortion in nanometres, 
        it must be multiplied by (wavelength / (2*pi)), (where `wavellength` here is the same wavelength
        in which r0 is given in the function arguments)
    """
    def set_X_coords(self):
        """
        Sets the coords of X, the new phase vector.
        """
        self.X_coords = numpy.zeros((self.nx_size, 2))
        self.X_coords[:, 0] = -1
        self.X_coords[:, 1] = numpy.arange(self.nx_size)
        self.X_positions = self.X_coords * self.pixel_scale

    def set_stencil_coords(self):
        """
        Sets the Z coordinates, sections of the phase screen that will be used to create new phase
        """
        self.stencil = numpy.zeros((self.stencil_length, self.nx_size))

        max_n = 1
        while True:
            if 2 ** (max_n - 1) + 1 >= self.nx_size:
                max_n -= 1
                break
            max_n += 1

        for n in range(0, max_n + 1):
            col = int((2 ** (n - 1)) + 1)
            n_points = (2 ** (max_n - n)) + 1

            coords = numpy.round(numpy.linspace(0, self.nx_size - 1, n_points)).astype('int32')
            self.stencil[col - 1][coords] = 1

        # Now fill in tail of stencil
        for n in range(1, self.stencil_length_factor + 1):
            col = n * self.nx_size - 1
            self.stencil[col, self.nx_size // 2] = 1

        self.stencil_coords = numpy.array(numpy.where(self.stencil == 1)).T
        self.stencil_positions = self.stencil_coords * self.pixel_scale

        self.n_stencils = len(self.stencil_coords)

    def calc_seperations(self):
        """
        Calculates the seperations between the phase points in the stencil and the new phase vector
        """
        positions = numpy.append(self.stencil_positions, self.X_positions, axis=0)
        self.seperations = numpy.zeros((len(positions), len(positions)))

        #if numba:
        #    calc_seperations_fast(positions, self.seperations)
        if 1: #else:
            for i, (x1, y1) in enumerate(positions):
                for j, (x2, y2) in enumerate(positions):
                    delta_x = x2 - x1
                    delta_y = y2 - y1

                    delta_r = numpy.sqrt(delta_x ** 2 + delta_y ** 2)

                    self.seperations[i, j] = delta_r



    def make_covmats(self):
        """
        Makes the covariance matrices required for adding new phase
        """
        self.cov_mat = phase_covariance(self.seperations, self.r0, self.L0)

        self.cov_mat_zz = self.cov_mat[:self.n_stencils, :self.n_stencils]
        self.cov_mat_xx = self.cov_mat[self.n_stencils:, self.n_stencils:]
        self.cov_mat_zx = self.cov_mat[:self.n_stencils, self.n_stencils:]
        self.cov_mat_xz = self.cov_mat[self.n_stencils:, :self.n_stencils]

    def makeAMatrix(self):
        """
        Calculates the "A" matrix, that uses the existing data to find a new 
        component of the new phase vector.
        """
        # Cholsky solve can fail - if so do brute force inversion
        try:
            cf = linalg.cho_factor(self.cov_mat_zz)
            inv_cov_zz = linalg.cho_solve(cf, numpy.identity(self.cov_mat_zz.shape[0]))
        except linalg.LinAlgError:
            # print("Cholesky solve failed. Performing SVD inversion...")
            # inv_cov_zz = numpy.linalg.pinv(self.cov_mat_zz)
            raise linalg.LinAlgError("Could not invert Covariance Matrix to for A and B Matrices. Try with a larger pixel scale or smaller L0")

        self.A_mat = self.cov_mat_xz.dot(inv_cov_zz)

    def makeBMatrix(self):
        """
        Calculates the "B" matrix, that turns a random vector into a component of the new phase.
        """
        # Can make initial BBt matrix first
        BBt = self.cov_mat_xx - self.A_mat.dot(self.cov_mat_zx)

        # Then do SVD to get B matrix
        u, W, ut = numpy.linalg.svd(BBt)

        L_mat = numpy.zeros((self.nx_size, self.nx_size))
        numpy.fill_diagonal(L_mat, numpy.sqrt(W))

        # Now use sqrt(eigenvalues) to get B matrix
        self.B_mat = u.dot(L_mat)

    def make_initial_screen(self):
        """
        Makes the initial screen usign FFT method that can be extended 
        """

        # phase screen will make it *really* random if no seed at all given.
        # If a seed is here, screen must be repeatable wiht same seed
        self._R = numpy.random.default_rng(self.random_seed)

        self._scrn = ft_phase_screen(
            self.r0, self.stencil_length, self.pixel_scale, self.L0, 1e-10, seed=self._R
        )

        self._scrn = self._scrn[:, :self.nx_size]

    def get_new_row(self):
        random_data = self._R.normal(0, 1, size=self.nx_size)

        stencil_data = self._scrn[(self.stencil_coords[:, 0], self.stencil_coords[:, 1])]
        new_row = self.A_mat.dot(stencil_data) + self.B_mat.dot(random_data)

        new_row.shape = (1, self.nx_size)
        return new_row

    def add_row(self):
        """
        Adds a new row to the phase screen and removes old ones.
        """

        new_row = self.get_new_row()

        self._scrn = numpy.append(new_row, self._scrn, axis=0)[:self.stencil_length, :self.nx_size]

        return self.scrn

    @property
    def scrn(self):
        """
        The current phase map held in the PhaseScreen object in radians.
        """
        return self._scrn[:self.requested_nx_size, :self.requested_nx_size]


class PhaseScreenVonKarman(PhaseScreen):
    """
    A "Phase Screen" for use in AO simulation with Von Karmon statistics.

    This represents the phase addition light experiences when passing through atmospheric
    turbulence. Unlike other phase screen generation techniques that translate a large static
    screen, this method keeps a small section of phase, and extends it as necessary for as many
    steps as required. This can significantly reduce memory consumption at the expense of more
    processing power required.

    The technique is described in a paper by Assemat and Wilson, 2006. It essentially assumes that
    there are two matrices, "A" and "B", that can be used to extend an existing phase screen.
    A single row or column of new phase can be represented by

        X = A.Z + B.b

    where X is the new phase vector, Z is some number of columns of the existing screen,
    and b is a random vector with gaussian statistics.

    This object calculates the A and B matrices using an expression of the phase covariance when it
    is initialised. Calculating A is straightforward through the relationship:

        A =  Cov_xz . (Cov_zz)^(-1).

    B is less trivial.

        BB^t = Cov_xx - A.Cov_zx

    (where B^t is the transpose of B) is a symmetric matrix, hence B can be expressed as

        B = UL,

    where U and L are obtained from the svd for BB^t

        U, w, U^t = svd(BB^t)

    L is a diagonal matrix where the diagonal elements are w^(1/2).

    On initialisation an initial phase screen is calculated using an FFT based method.
    When ``add_row`` is called, a new vector of phase is added to the phase screen using `nCols`
    columns of previous phase. Assemat & Wilson claim that two columns are adequate for good
    atmospheric statistics. The phase in the screen data is always accessed as ``<phasescreen>.scrn`` and is in radians.

        .. note::
        The phase screen is returned on each iteration as a 2d array, with each element representing the phase 
        change in **radians**. This means that to obtain the physical phase distortion in nanometres, 
        it must be multiplied by (wavelength / (2*pi)), (where `wavellength` here is the same wavelength
        in which r0 is given in the function arguments)

    Parameters:
        nx_size (int): Size of phase screen (NxN)
        pixel_scale(float): Size of each phase pixel in metres
        r0 (float): fried parameter (metres)
        L0 (float): Outer scale (metres)
        random_seed (int, optional): seed for the random number generator
        n_columns (int, optional): Number of columns to use to continue screen, default is 2
    """
    def __init__(self, nx_size, pixel_scale, r0, L0, random_seed=None, n_columns=2):

        self.n_columns = n_columns

        self.requested_nx_size = nx_size
        self.nx_size = nx_size
        self.pixel_scale = pixel_scale
        self.r0 = r0
        self.L0 = L0
        self.stencil_length_factor = 1
        self.stencil_length = self.nx_size

        self.random_seed = random_seed

        self.set_X_coords()
        self.set_stencil_coords()

        self.calc_seperations()
        self.make_covmats()

        self.makeAMatrix()
        self.makeBMatrix()
        self.make_initial_screen()


    def set_stencil_coords(self):
        self.stencil = numpy.zeros((self.stencil_length, self.nx_size))
        self.stencil[:self.n_columns] = 1

        self.stencil_coords = numpy.array(numpy.where(self.stencil==1)).T
        self.stencil_positions = self.stencil_coords * self.pixel_scale

        self.n_stencils = len(self.stencil_coords)


def find_allowed_size(nx_size):
    """
    Finds the next largest "allowed size" for the Fried Phase Screen method
    
    Parameters:
        nx_size (int): Requested size
    
    Returns:
        int: Next allowed size
    """
    n = 0
    while (2 ** n + 1) < nx_size:
        n += 1

    nx_size = 2 ** n + 1
    return nx_size


class PhaseScreenKolmogorov(PhaseScreen):
    """
    A "Phase Screen" for use in AO simulation using the Fried method for Kolmogorov turbulence.

    This represents the phase addition light experiences when passing through atmospheric 
    turbulence. Unlike other phase screen generation techniques that translate a large static 
    screen, this method keeps a small section of phase, and extends it as neccessary for as many 
    steps as required. This can significantly reduce memory consuption at the expense of more 
    processing power required.

    The technique is described in a paper by Assemat and Wilson, 2006 and expanded upon by Fried, 2008.
    It essentially assumes that there are two matrices, "A" and "B",
    that can be used to extend an existing phase screen.
    A single row or column of new phase can be represented by 

        X = A.Z + B.b

    where X is the new phase vector, Z is some data from the existing screen,
    and b is a random vector with gaussian statistics.

    This object calculates the A and B matrices using an expression of the phase covariance when it
    is initialised. Calculating A is straightforward through the relationship:

        A =  Cov_xz . (Cov_zz)^(-1).

    B is less trivial.

        BB^t = Cov_xx - A.Cov_zx

    (where B^t is the transpose of B) is a symmetric matrix, hence B can be expressed as 

        B = UL, 

    where U and L are obtained from the svd for BB^t

        U, w, U^t = svd(BB^t)

    L is a diagonal matrix where the diagonal elements are w^(1/2).    

    The Z data is taken from points in a "stencil" defined by Fried that samples the entire screen.
    An additional "reference point" is also considered, that is picked from a point separate from teh stencil 
    and applied on each iteration such that the new phase equation becomes:
    
    On initialisation an initial phase screen is calculated using an FFT based method.
    When ``add_row`` is called, a new vector of phase is added to the phase screen. The phase in the screen data
    is always accessed as ``<phasescreen>.scrn`` and is in radians.

    .. note::
        The phase screen is returned on each iteration as a 2d array, with each element representing the phase 
        change in **radians**. This means that to obtain the physical phase distortion in nanometres, 
        it must be multiplied by (wavelength / (2*pi)), (where `wavellength` here is the same wavelength
        in which r0 is given in the function arguments)

    Parameters:
        nx_size (int): Size of phase screen (NxN)
        pixel_scale(float): Size of each phase pixel in metres
        r0 (float): fried parameter (metres)
        L0 (float): Outer scale (metres)
        random_seed (int, optional): seed for the random number generator
        stencil_length_factor (int, optional): How much longer is the stencil than the desired phase? default is 4
    """
    def __init__(self, nx_size, pixel_scale, r0, L0, random_seed=None, stencil_length_factor=4):

        self.requested_nx_size = nx_size
        self.nx_size = find_allowed_size(nx_size)
        self.pixel_scale = pixel_scale
        self.r0 = r0
        self.L0 = L0
        self.stencil_length_factor = stencil_length_factor
        self.stencil_length = stencil_length_factor * self.nx_size
        self.random_seed = random_seed

        # Coordinate of Fried's "reference point" that stops the screen diverging
        self.reference_coord = (1, 1)

        self.set_X_coords()
        self.set_stencil_coords()

        self.calc_seperations()
        self.make_covmats()

        self.makeAMatrix()
        self.makeBMatrix()
        self.make_initial_screen()

    def get_new_row(self):
        random_data = self._R.normal(0, 1, size=self.nx_size)

        stencil_data = self._scrn[(self.stencil_coords[:, 0], self.stencil_coords[:, 1])]

        reference_value = self._scrn[self.reference_coord]

        new_row = self.A_mat.dot(stencil_data - reference_value) + self.B_mat.dot(random_data) + reference_value

        new_row.shape = (1, self.nx_size)
        return new_row


    def __repr__(self):
        return str(self.scrn)
    


#@numba.jit(nopython=True, parallel=True)
# def calc_seperations_fast(positions, seperations):

#     for i in numba.prange(len(positions)):
#         x1, y1 = positions[i]
#         for j in range(len(positions)):
#             x2, y2 = positions[j]
#             delta_x = x2 - x1
#             delta_y = y2 - y1

#             delta_r = numpy.sqrt(delta_x ** 2 + delta_y ** 2)

#             seperations[i, j] = delta_r