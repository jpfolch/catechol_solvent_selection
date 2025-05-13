class graph:
    def __init__(self, F, A, L=4, y=None):
        self.N = len(F)
        self.M = len(F[1])
        self.F = F
        self.A = A
        self.L = L
        self.y = y
        self.S = [sum(x) for x in zip(*self.F)]
        self.l = [0] * self.N
        for v in range(self.N):
            for l in range(self.L):
                if self.F[v][l]:
                    self.l[v] = l
                    break
        self.calculate_shortest_distance()

    def calculate_shortest_distance(self):
        self.dis = []
        for i in range(self.N):
            self.dis.append([])
            for j in range(self.N):
                self.dis[i].append(self.N)
                if self.A[i][j]:
                    self.dis[i][j] = 1
            self.dis[i][i] = 0
        for k in range(self.N):
            for i in range(self.N):
                for j in range(self.N):
                    self.dis[i][j] = min(
                        self.dis[i][j], self.dis[i][k] + self.dis[k][j]
                    )
        self.D = {}
        self.P = {}
        for s in range(self.N):
            self.D[s] = 0
            for l1 in range(self.L):
                for l2 in range(self.L):
                    self.P[(s, l1, l2)] = 0
        for u in range(self.N):
            for v in range(self.N):
                if self.dis[u][v] == self.N:
                    continue
                self.D[self.dis[u][v]] += 1
                self.P[(self.dis[u][v], self.l[u], self.l[v])] += 1
