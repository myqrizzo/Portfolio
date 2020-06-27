class Matrix:
    def __init__(self, matrix):
        self.matrix = matrix

    def row(self, index):
        return self.matrix[index]

    def column(self, index):
        #print(self.matrix, index)
        return [row[index] for row in self.matrix]

    def rowSize(self):
        return len(self.row(0))

    def colSize(self):
        return len(self.column(0))


def saddle_points(matrix):
    matA = Matrix(matrix)
    saddlepoints = []
    if len(matA.matrix) > 0:
        for i in range(matA.colSize()):
            # first get a copy of the current row
            row_i = matA.row(i).copy()
            # check to see if matrix is regular
            if len(row_i) != matA.rowSize():
                raise ValueError("This matrix is irregular.")
            indexR = i+1
            # search for max values within a row, it may be possible that there are repeats
            row_size = len(row_i)
            column_indices = []
            largest = -100
            for ind in range(row_size):
                curr_largest = max(row_i)
                if curr_largest >= largest:
                    largest = curr_largest
                    column_indices.append(row_i.index(
                        largest)+len(column_indices))
                    row_i.remove(curr_largest)
            # check if a max value corresponds to a min value in a column
            col_ind_size = len(column_indices)
            for ind in range(col_ind_size):
                if largest == min(matA.column(column_indices[ind])):
                    # if yes, append as a saddlepoint
                    saddlepoints.append(
                        {'row': indexR, 'column': column_indices[ind]+1})
    return saddlepoints
