from megaman.relaxation.precomputed import *
from .utils import generate_toy_laplacian

class BaseTestARkNeighbors(object):
    def generate_laplacian(self):
        raise NotImplementedError()
    def setup_message(self):
        raise NotImplementedError()

    def setUp(self):
        self.generate_laplacian_and_range()
        self.setup_message()
        self.A, self.pairs = makeA(self.laplacian)

        # HACK: A is somehow sorted by column, so here I'll change it manually.
        sortbyrow = np.lexsort((self.pairs[:,1],self.pairs[:,0]))
        self.A = self.A[sortbyrow]
        self.pairs = self.pairs[sortbyrow]

        # self.Rk_tensor, self.nbk = compute_Rk(self.laplacian,self.A,self.n)
        self.correct_S, self.correct_pairs = self.project_S_from_laplacian()

    def generate_laplacian_and_range(self):
        self.laplacian = self.generate_laplacian()
        self.n = self.laplacian.shape[0]
        self.range = np.arange(self.n)
        self.Y = self.generate_toy_Y()

    def generate_toy_Y(self):
        return np.random.uniform(size=self.n)

    def ij_is_neighbors(self,i,j):
        return self.laplacian[i,j] != 0

    def project_S_from_laplacian(self):
        # TODO: make the test process faster!
        S = [ self.Y[i]-self.Y[j] for i in np.arange(self.n) \
              for j in np.arange(i+1,self.n) \
              if self.ij_is_neighbors(i,j) ]
        pairs = [ [i,j] for i in np.arange(self.n) \
                  for j in np.arange(i+1,self.n) \
                  if self.ij_is_neighbors(i,j) ]
        return np.array(S), np.array(pairs)

    def test_A_length_equality(self):
        A_length = self.A.shape[0]
        correct_A_length = self.correct_S.shape[0]
        assert A_length == correct_A_length, 'The first dimension of A is calculated wrong.'

    def test_pairs(self):
        np.testing.assert_array_equal(
            self.pairs, self.correct_pairs,
            err_msg='Sorted pairs should be the same.'
        )

    def test_A(self):
        testing_S = self.A.dot(self.Y)
        np.testing.assert_allclose(
            testing_S, self.correct_S,
            err_msg='A*y should be the same as yj-yi for all j>i'
        )

    def _test_ATAinv(self):
        # TODO: why this test will running out of the memory?
        ATAinv = np.linalg.pinv(self.A.T.dot(self.A).todense())
        S = self.A.dot(self.Y)
        testing_Y = ATAinv.dot(self.A.T).dot(S)
        np.testing.assert_allclose(
            testing_Y, self.Y,
            err_msg='ATAinv * AT * S should be the same as original Y'
        )

    def _test_Rk(self):
        # TODO: Need to understand what Rk means.
        pass

class TestAkRkNbkFromToyLaplacian(BaseTestARkNeighbors):
    def generate_laplacian(self):
        return generate_toy_laplacian(n=200)
    def setup_message(self):
        print ('Tesking Rk properties for toy laplacian.')
