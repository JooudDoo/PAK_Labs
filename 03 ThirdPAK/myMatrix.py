class Matrix:
    POSSIBLE_TYPES = [int, float]
    data = []

    rows = 0
    columns = 0

    def __init__(self, rows : 'list[list]' = [[]], cntrows = 2, cntcolumns = 2):
        if(rows != [[]]):
            self.check_integrity(rows)
            self.data = rows
            self.update_rows_columns()
        else: #Set matrix as empty
            self.set_matrix(rows=[[0 for collumn in range(cntcolumns)] for row in range(cntrows)])
    
    def __add__(self, other):
        return self.add(other)

    def __mul__(self, other):
        return self.mult(other)
    
    def __sub__(self, other):
        return self.sub(other)

    def __getitem__(self, key):
        return self.data[key]

    def __repr__(self):
        res = ""
        for row in self.data:
            res += str(row) + "\n"
        return res

    def set_matrix(self, rows : list[list]):
        self.check_integrity(rows)
        self.data = rows
        self.update_rows_columns()

    def write_matrix_to_file(self, file):
        with open(file, "w") as f:
            f.write(str(self))
    
    def update_rows_columns(self):
        self.rows = len(self.data)
        self.columns = len(self[0])

    def transpose(self):
        matr = Matrix(cntrows=self.columns, cntcolumns=self.rows)
        for row in range(self.rows):
            for column in range(self.columns):
                matr[column][row] = self[row][column]
        return matr

    def sub(self, other_matr):
        if not(self.rows == other_matr.rows and self.columns == other_matr.columns):
            raise Exception(f"Can't sub ({self.rows}, {self.columns}) matrix with ({other_matr.rows}, {other_matr.columns}) matrix")
        newmatr = Matrix(cntrows=self.rows, cntcolumns=other_matr.columns)
        for row in range(self.rows):
            for column in range(self.columns):
                newmatr[row][column] += self[row][column] - other_matr[row][column]
        return newmatr
    
    def add(self, other_matr):
        if not(self.rows == other_matr.rows and self.columns == other_matr.columns):
            raise Exception(f"Can't add ({self.rows}, {self.columns}) matrix with ({other_matr.rows}, {other_matr.columns}) matrix")
        newmatr = Matrix(cntrows=self.rows, cntcolumns=other_matr.columns)
        for row in range(self.rows):
            for column in range(self.columns):
                newmatr[row][column] += self[row][column] + other_matr[row][column]
        return newmatr

    def mult(self, other_matr):
        if not (self.rows == other_matr.columns):
            raise Exception(f"Can't multiply ({self.rows}, {self.columns}) matrix with ({other_matr.rows}, {other_matr.columns}) matrix")
        newmatr = Matrix(cntrows=self.rows, cntcolumns=other_matr.columns)
        matrT = other_matr.transpose()
        for row in range(self.rows):
            for row_t in range(matrT.rows):
                elem = 0
                for column in range(self.columns):
                    elem += self[row][column] * matrT[row_t][column]
                newmatr[row][row_t] = elem
        return newmatr
    
    def check_integrity(self, rows : list[list] = []):
        if(rows == []): rows = self.data
        if(rows == []): return
        first_row_len = len(rows[0])
        for row in rows:
            if(first_row_len != len(row)):
                raise Exception(f"Not correct matrix")
            for column in row:
                if(type(column) not in self.POSSIBLE_TYPES):
                    raise TypeError(f"Incorrect type for matrix {type(column)}")
    
    def matrix_fold(self, other):
        newmatr = Matrix(cntrows=other.rows, cntcolumns=other.columns)
        for ver_shift in range(self.columns-other.columns + 1):
            for hor_shift in range(self.rows-other.rows + 1):
                for row in range(other.rows):
                    for column in range(other.columns):
                        newmatr[ver_shift][hor_shift] += self[ver_shift+row][hor_shift+column] * other[row][column]
        return newmatr


def read_matrixs_from_file(file, convert_type = int) -> list[Matrix]:
    if(convert_type not in Matrix.POSSIBLE_TYPES):
        raise TypeError(f"Incorrect type for convert {convert_type}")
    matrixs : list[Matrix] = []
    matr = []
    cnt_readed = 0
    f = open(file, 'r')
    while True:
        line = f.readline()
        if(line == '\n' or line == ''):
            cnt_readed += 1
            matrixs.append(Matrix(matr))
            matr = []
            if(line == ''): break
        else:
            matr.append([convert_type(x) for x in line.split(' ') if x != ''])
    f.close()
    return matrixs