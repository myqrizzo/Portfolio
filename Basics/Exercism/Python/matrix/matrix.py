class Matrix:
    def __init__(self, matrix_string):
        matrix_temp = matrix_string.split("\n")
        self.matrix = [list(map(int, entry.split(" ")))
                       for entry in matrix_temp]

    def row(self, index):
        return self.matrix[index-1]

    def column(self, index):
        return [row[index-1] for row in self.matrix]
