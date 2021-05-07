class Grammar:
    def __init__(self):
        self.Rule = 0
        self.Antecedent = 1
        self.Consequent = 2
        self.Comparison = 3
        self.CategoricalComparator = 4
        self.CategoricalAttributeComparison = 5
        self.NumericalComparator = 6
        self.NumericalAttributeComparison = 7
        self.NonTerminal = [self.Rule,self.Antecedent ,self.Consequent,self.Comparison,self.CategoricalComparator,
                            self.CategoricalAttributeComparison,self.NumericalComparator,self.NumericalAttributeComparison]
        self.AND = 8
        self.Different = 9
        self.Equal = 10
        self.InferiorOrEqual = 11
        self.Inferior = 12
        self.SuperiorOrEqual = 13
        self.Superior = 14
        self.Name = 15
        self.Value = 16
        self.Terminal = [self.AND,self.Different,self.Equal,self.InferiorOrEqual,self.Inferior,self.SuperiorOrEqual,self.Superior,self.Name,self.Value]


