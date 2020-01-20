# Rede Neural de Multicamadas
 Uma melhora na rede neural simples, adicionando mais camadas e dificultando o problema

Ha mais ou menos um ano atrás fiz um fork de uma rede neural simples e melhorei o código, atualizei para o Python 3, traduzi e postei, acabei aprendendo como criar uma rede neural simples para resolver um problema de prever números em uma tabela, agora com um problema mais difícil na qual aquela rede neural não conseguiria resolver.

# O problema

![Image description](https://miro.medium.com/max/682/1*WlCd6kgxqb2yQpcutLhYwQ.png)

O truque é perceber que a terceira coluna é irrelevante, mas as duas primeiras colunas exibem o comportamento de uma porta XOR . Se a primeira coluna ou a segunda coluna for 1, a saída será 1. No entanto, se as duas colunas forem 0 ou as duas colunas forem 1, a saída será 0.
Portanto, a resposta correta é 0.
No entanto, isso seria demais para o nosso único neurônio suportar. Isso é considerado um "padrão não linear", porque não há relação direta direta entre as entradas e a saída.
Em vez disso, devemos criar uma camada oculta adicional, composta por quatro neurônios (Camada 1). Essa camada permite que a rede neural pense em combinações de entradas.

![Image description](https://miro.medium.com/max/423/1*Qt5lealRQ29-R8rcTPDtoA.png)

Você pode ver no diagrama que a saída da camada 1 é alimentada na camada 2. Agora é possível para a rede neural descobrir correlações entre a saída da camada 1 e a saída no conjunto de treinamento. Conforme a rede neural aprende, amplificará essas correlações ajustando os pesos nas duas camadas.
De fato, o reconhecimento de imagem é muito semelhante. Não há relação direta entre pixels e maçãs. Mas há uma relação direta entre combinações de pixels e maçãs.

![Image description](https://miro.medium.com/max/160/1*YqjgIOW86wioEhmZeWbcqw.jpeg)

O processo de adicionar mais camadas a uma rede neural, para que ela possa pensar em combinações, é chamado de "aprendizado profundo".Ok, estamos prontos para o código Python? Primeiro vou dar o código e depois vou explicar mais.

# Aqui estaria o código, você pode olhar dentro do arquivo "aprendizado_profundo.py"

O que é diferente desta vez é que existem várias camadas. Quando a rede neural calcula o erro na camada 2, ela propaga o erro para trás na camada 1, ajustando os pesos à medida que avança. Isso é chamado de "propagação traseira".

![Image description](https://miro.medium.com/max/840/1*-1trgA6DUEaafJZv3k0mGw.jpeg)

# Dependências

"""
pip install numpy
python3 main.py
"""

Créditos: Milo Spencer-Harper

# Alguns Links

- https://medium.com/technology-invention-and-more/how-to-build-a-multi-layered-neural-network-in-python-53ec3d1d326a
- https://github.com/miloharper/multi-layer-neural-network

> Eu acabei não fazendo um fork por um simples motivo: acabei criando um repositório e esqueci de fazer um fork
