import numpy as np
from numpy import linalg as la


np.set_printoptions(precision=3)



class DiGraph:
    """A class for representing directed graphs via their adjacency matrices.

    Attributes:
        A_hat
    """
    # Problem 1
    def __init__(self, A, labels=None):
        """Modify A so that there are no sinks in the corresponding graph,
        then calculate Ahat. Save Ahat and the labels as attributes.

        Parameters:
            A ((n,n) ndarray): the adjacency matrix of a directed graph.
                A[i,j] is the weight of the edge from node j to node i.
            labels (list(str)): labels for the n nodes in the graph.
                If None, defaults to [0, 1, ..., n-1].

        Examples
        ========
        >>> A = np.array([[0, 0, 0, 0],[1, 0, 1, 0],[1, 0, 0, 1],[1, 0, 1, 0]])
        >>> G = DiGraph(A, labels=['a','b','c','d'])
        >>> G.A_hat
        array([[0.   , 0.25 , 0.   , 0.   ],
               [0.333, 0.25 , 0.5  , 0.   ],
               [0.333, 0.25 , 0.   , 1.   ],
               [0.333, 0.25 , 0.5  , 0.   ]])
        >>> steady_state_1 = G.linsolve()
        >>> { k: round(steady_state_1[k],3) for k in steady_state_1}
        {'a': 0.096, 'b': 0.274, 'c': 0.356, 'd': 0.274}
        >>> steady_state_2 = G.eigensolve()
        >>> { k: round(steady_state_2[k],3) for k in steady_state_2}
        {'a': 0.096, 'b': 0.274, 'c': 0.356, 'd': 0.274}
        >>> steady_state_3 = G.itersolve()
        >>> { k: round(steady_state_3[k],3) for k in steady_state_3}
        {'a': 0.096, 'b': 0.274, 'c': 0.356, 'd': 0.274}
        >>> get_ranks(steady_state_3)
        ['c', 'b', 'd', 'a']
        """

        #Handling labels
        size = A.shape[0]
        if (labels == None):
            l = np.arange(size)
            labels = map(str, l)
        elif (size != len(labels)):
            raise ValueError("length of label array is not the same as the matrix size")
        self.labels = labels

        #getting A_wave
        for i in range(size):
            c = A[:,i]
            if (np.sum(c)==0):
                A[:,i] = 1
        #getting A_hat
        hat = np.array([[0.1 for x in range(size)] for x in range(size)])
        for i in range(size):
            c = A[:,i]
            c = c/np.sum(c)
            hat[:,i]=c

        self.A_hat = hat



    def linsolve(self, epsilon=0.85):
        """Compute the PageRank vector using the linear system method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Returns:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        size = self.A_hat.shape[0]
        a = np.identity(size)-(epsilon*self.A_hat)
        # b = np.array([[1.0 for x in range(size)] for x in range(size)])
        b = np.array([1.0 for x in range(size)])
        b = b*((1-epsilon)/size)
        x = np.linalg.tensorsolve(a, b)
        d = dict()
        for i in range(size):
            d.update({self.labels[i] : x[i]})
        return d


    def eigensolve(self, epsilon=0.85):
        """Compute the PageRank vector using the eigenvalue method.
        Normalize the resulting eigenvector so its entries sum to 1.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        pass


    def itersolve(self, epsilon=0.85, maxiter=100, tol=1e-12):
        """Compute the PageRank vector using the iterative method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.
            maxiter (int): the maximum number of iterations to compute.
            tol (float): the convergence tolerance.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        size = self.A_hat.shape[0]
        x =  np.array([1.0/size for x in range(size)])
        x_old = np.array([0.0/size for x in range(size)])
        t = 0
        ones = np.array([[1.0 for x in range(size)] for x in range(size)])
        while ((x - x_old).all() > tol and t < maxiter):
            x_old = x
            x = np.dot((epsilon * self.A_hat) + ((1-epsilon)*(ones/size)), x)
            t = t + 1
        d = dict()
        for i in range(size):
            d.update({self.labels[i] : x[i]})
        return d



def get_ranks(d):
    """Construct a sorted list of labels based on the PageRank vector.

    Parameters:
        d (dict(str -> float)): a dictionary mapping labels to PageRank values.

    Returns:
        (list) the keys of d, sorted by PageRank value from greatest to least.
    """
    return sorted(d, key=d.__getitem__, reverse=True)


# Task 2
def rank_websites(filename="web_stanford.txt", epsilon=0.85):
    """Read the specified file and construct a graph where node j points to
    node i if webpage j has a hyperlink to webpage i. Use the DiGraph class
    and its itersolve() method to compute the PageRank values of the webpages,
    then rank them with get_ranks().

    Each line of the file has the format
        a/b/c/d/e/f...
    meaning the webpage with ID 'a' has hyperlinks to the webpages with IDs
    'b', 'c', 'd', and so on.

    Parameters:
        filename (str): the file to read from.
        epsilon (float): the damping factor, between 0 and 1.

    Returns:
        (list(str)): The ranked list of webpage IDs.

    Examples
    ========
    >>> print(rank_websites()[0:5])
    ['98595', '32791', '28392', '77323', '92715']
    """
    #reading file and splitting at new line
    file = open(filename)
    file_string = file.read()
    file_array = file_string.split('\n')

    #Creating a dictionary, mapping a to ['b','c', 'd'] for the line a/b/c/d
    d = dict()
    for i in range(len(file_array)):
        if (file_array[i]==''):
            continue
        line_array = file_array[i].split('/')
        d.update({line_array[0] : line_array[1:]})

    #Creating a list of labels and the matrix A for DiGraph
    labels = list(d.keys())
    A = np.array([[0 for x in range(len(labels))] for x in range(len(labels))])
    c = np.array([0 for x in range(len(labels))])
    for i in range(len(labels)):
        c.fill(0)
        v = d[labels[i]]
        for j in range(len(labels)):
            if (labels[j] in v):
                c[j] = 1
        A[:,i] = c
    G = DiGraph(A, labels)
    d = G.itersolve(epsilon)

    return get_ranks(d)


# Task 3
def rank_uefa_teams(filename, epsilon=0.85):
    """Read the specified file and construct a graph where node j points to
    node i with weight w if team j was defeated by team i in w games. Use the
    DiGraph class and its itersolve() method to compute the PageRank values of
    the teams, then rank them with get_ranks().

    Each line of the file has the format
        A,B
    meaning team A defeated team B.

    Parameters:
        filename (str): the name of the data file to read.
        epsilon (float): the damping factor, between 0 and 1.

    Returns:
        (list(str)): The ranked list of team names.

    Examples
    ========
    >>> rank_uefa_teams("psh-uefa-2018-2019.csv",0.85)[0:5]
    ['Liverpool', 'Ath Madrid', 'Paris SG', 'Genk', 'Barcelona']
    """
    #reading file and splitting at new line
    file = open(filename)
    file_string = file.read()
    matches = file_string.split('\n')
    #Creating a dictionary mapping teams to a list of teams they have lost to
    d = dict()
    for i in range(len(matches)):
        if (matches[i]==''):
            continue
        mi = matches[i].split(',')
        if (mi[0] not in d.keys()):
            d.update({mi[0] : np.array([])})
        if (mi[1] not in d.keys()):
            d.update({mi[1] : np.array([])})

        if (mi[2] > mi[3]):
            d[mi[1]] = np.append(d[mi[1]], mi[0])
        if (mi[3] > mi[2]):
            d[mi[0]] = np.append(d[mi[0]], mi[1])

    #Creating a list of labels and the matrix A for DiGraph
    labels = list(d.keys())
    A = np.array([[0 for x in range(len(labels))] for x in range(len(labels))])
    c = np.array([0 for x in range(len(labels))])
    for i in range(len(labels)):
        c.fill(0)
        v = d[labels[i]]
        for j in range(len(labels)):
            if (labels[j] in v):
                c[j] = list(v).count(labels[j])
        A[:,i] = c

    G = DiGraph(A, labels)
    d = G.itersolve(epsilon)

    return get_ranks(d)






if __name__ == "__main__":
    import doctest
    doctest.testmod()
