'''
cv2.threshold(): 
    Aplica limiarização global à imagem, compara cada pixel a um único valor fixo de limiar.

    Sintaxe: retval, dst = cv2.threshold(src, thresh, maxval, type)

    Com:    src: imagem de entrada em tons de cinza
            thresh: valor do limiar
            maxval: valor que será atribuído aos pixels acima do limiar
            type: tipo de limiarização:
                    cv2.THRESH_BINARY: abaixo do limiar vira 0, acima vira maxval
                    cv2.THRESH_BINARY_INV: inverte o acima/abaixo
                    cv2.THRESH_TRUNC: pixels acima do limiar são "cortados" no limiar
                    cv2.THRESH_TOZERO: abaixo do limiar vira 0; acima permanece
                    cv2.THRESH_TOZERO_INV: inverso do anterior

cv2.adaptiveThreshold():
    Aplica limiarização adaptativa: o limiar muda de acordo com pequenas regiões da imagem (útil para imagens com iluminação não uniforme).

    Sintaxe: dst = cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C)

    Com:    src: imagem em tons de cinza
            maxValue: valor atribuído a pixels que passam no teste
            adaptiveMethod: método para calcular o limiar de cada região:
                cv2.ADAPTIVE_THRESH_MEAN_C: média dos vizinhos menos C
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C: média ponderada (Gaussiana) dos vizinhos menos C
    thresholdType: geralmente cv2.THRESH_BINARY ou cv2.THRESH_BINARY_INV
    blockSize: tamanho da vizinhança (deve ser ímpar, ex: 11)
    C: constante subtraída da média ou da média ponderada   

'''
