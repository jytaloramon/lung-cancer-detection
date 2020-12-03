# Lung Cancer Detection (Detecção de Cancer de Pulmão)

**Universidade Federal de Alagoas - UFAL**

**Programa Institucional de Bolsa de Iniciação Científica - PIBIC**

**Título**: Auxílio Computadorizado à Identificação de Nódulos Pulmonares Baseado em Técnicas de Aprendizado Profundo

**Integrantes:**
- J.Y. Ramon A. Gonçalves (Processamento de imagem)
- Eduardo Silva (Machine Learning)
- [**Rodolfo Carneiro Cavalcante** (Orientador)](http://buscatextual.cnpq.br/buscatextual/visualizacv.do?id=K4480200Y9)

## **Sobre**

O câncer de pulmão é o tipo de câncer que mais causa mortes no mundo e sua principal manifestação ocorre devido ao aparecimento de lesões no tecido pulmonar. Dada a sua taxa de mortalidade, é muito importante que esse tipo de câncer seja identificado e diagnosticado o mais rápido possível. Diante disso, esse tipo de câncer vem sendo cada vez mais estudado e na tentativa de automatizar o processo de detecção – utilizando métodos de aprendizagem de máquina – comumente surgi alguns desafios – estando relacionados com a qualidade das imagens radiográficas –, sendo os principais: a quantidade de imagens, problemas com ruídos e distorções, além das regiões de interesse (ROIs) que são relativamente pequenas. Por isso, é de suma importância reduzir as áreas de buscas por possíveis nódulos, removendo ruídos e, consequentemente reduzindo as chances de erros de predição feitas pelas redes de Machine Learning. [Processamento de Imagem]

## **Processamento de imagem**

**Dados (data):** os subsets estão no formato array, contendo tuplas no formato: *tuple(matrix_pixel, array_points_nodules[tuple(axis_y, axis_x)])*. Os dados da matriz não estão normalizados.

[**lung_roi.py:**](lung_roi.py) arquivo destinado para segmentação da imagem e extração das regiões de interesse - pulmão - (ROI).

</br>

### **Resultados da Segmentação**

</br>

<p align="center">
  <img src="./images/fig_1.png" alt='Fig_1' width='360px' height='240px'>
  <img src="./images/fig_2.png" alt='Fig_2' width='360px' height='240px'>
  <img src="./images/fig_4.png" alt='Fig_4' width='360px' height='240px'>
  <img src="./images/fig_5.png" alt='Fig_5' width='360px' height='240px'>
  </br>
  <i>Processo de Segmentação: (1) imagem de entrada, (2) imagem com filtro Gaussiano, (3) máscara gerada, (4) extração.</i>
</p>

```Os teste foram feitos com 894 imagens, onde apenas 16 apresentaram problemas na segmentação. Dessas 16 imagens apenas 11 delas apresentou interferência na região do nódulo.``` 

