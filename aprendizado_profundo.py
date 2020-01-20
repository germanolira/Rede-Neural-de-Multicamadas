from numpy import exp, array, random, dot


class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1


class NeuralNetwork():
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2

    # A função Sigmoid, que descreve uma curva em forma de S.
    # Passamos a soma ponderada das entradas por essa função para
    # normalize-os entre 0 e 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # A derivada da função Sigmoide.
    # Este é o gradiente da curva sigmóide.
    # Indica a confiança que temos sobre o peso existente.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # Treinamos a rede neural através de um processo de tentativa e erro.
    # Ajustar os pesos sinápticos de cada vez.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # Passe o conjunto de treinamento através da nossa rede neural
            output_from_layer_1, output_from_layer_2 = self.think(training_set_inputs)

            # Calcule o erro da camada 2 (a diferença entre a saída desejada
            # e a saída prevista).
            layer2_error = training_set_outputs - output_from_layer_2
            layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2)

            # Calcule o erro da camada 1 (observando os pesos na camada 1,
            # podemos determinar por quanto a camada 1 contribuiu para o erro na camada 2).
            layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
            layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)

            # Calcule quanto ajustar os pesos,
            layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)

            # Ajuste os pesos.
            self.layer1.synaptic_weights += layer1_adjustment
            self.layer2.synaptic_weights += layer2_adjustment

    # A rede neural pensa.
    def think(self, inputs):
        output_from_layer1 = self.__sigmoid(dot(inputs, self.layer1.synaptic_weights))
        output_from_layer2 = self.__sigmoid(dot(output_from_layer1, self.layer2.synaptic_weights))
        return output_from_layer1, output_from_layer2

    # A rede neural imprime seus pesos
    def print_weights(self):
        print ("    Layer 1 (4 neurons, each with 3 inputs): ")
        print (self.layer1.synaptic_weights)
        print ("    Layer 2 (1 neuron, with 4 inputs):")
        print (self.layer2.synaptic_weights)

if __name__ == "__main__":

   #Seed o gerador de números aleatórios
    random.seed(1)

    # Crie a camada 1 (4 neurônios, cada um com 3 entradas)
    layer1 = NeuronLayer(4, 3)

    # Crie a camada 2 (um único neurônio com 4 entradas)
    layer2 = NeuronLayer(1, 4)

    # Combine as camadas para criar uma rede neural
    neural_network = NeuralNetwork(layer1, layer2)

    print ("Stage 1) Random starting synaptic weights: ")
    neural_network.print_weights()

    # O conjunto de treinamento. Temos 7 exemplos, cada um composto por 3 valores de entrada
    # e 1 valor de saída.
    training_set_inputs = array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
    training_set_outputs = array([[0, 1, 1, 1, 1, 0, 0]]).T

    # Treine a rede neural usando o conjunto de treinamento.
    # Faça 60.000 vezes e faça pequenos ajustes a cada vez.
    neural_network.train(training_set_inputs, training_set_outputs, 60000)

    print ("Stage 2) New synaptic weights after training: ")
    neural_network.print_weights()

    # Teste a rede neural com uma nova situação.
    print ("Stage 3) Considering a new situation [1, 1, 0] -> ?: ")
    hidden_state, output = neural_network.think(array([1, 1, 0]))
    print (output)