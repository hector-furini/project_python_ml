`No treino da classificação a maioria dos modelos apresentou métricas muito perfeitas. Porque isso acontece?`

Isso acontece por uma combinação de **três fatores**: o funcionamento interno da classe [`Classifier`](file:///home/hector_furini/Projeto_ML_Python/projeto_hector/dev/script/projeto_hector/src/classifier.py#L43-L570), a proporção entre o número de variáveis e o número de pacientes, e os hiperparâmetros que o `GridSearchCV` escolheu.

Na prática, os modelos **não aprenderam um padrão biológico perfeito — eles decoraram (*overfitting*) as amostras**.

---

### 1. O método `classify()` treina e avalia nos mesmos dados
Nos métodos internos da classe [`Classifier`](file:///home/hector_furini/Projeto_ML_Python/projeto_hector/dev/script/projeto_hector/src/classifier.py#L65-L153) ([`__matrix_confusao`](file:///home/hector_furini/Projeto_ML_Python/projeto_hector/dev/script/projeto_hector/src/classifier.py#L65-L91), [`__retornar_metricas`](file:///home/hector_furini/Projeto_ML_Python/projeto_hector/dev/script/projeto_hector/src/classifier.py#L93-L123) e [`__metricas_pontuais`](file:///home/hector_furini/Projeto_ML_Python/projeto_hector/dev/script/projeto_hector/src/classifier.py#L125-L153)), o código faz exatamente isto:

```python
grid_search.fit(self.X, self.y)          # 1. Ajusta o modelo nos dados passados
best_model = grid_search.best_estimator_ #    (por padrão refit=True no dataset inteiro)
y_pred = best_model.predict(self.X)      # 2. Faz a predição NOS MESMOS dados!
```

Ou seja:
* Na **Seção 2**, quando você passa `df_train` (131 pacientes), `df_valid` (44 pacientes) ou `df_test` (44 pacientes) para o `Classifier(df=...)`, ele treina o melhor modelo naquele conjunto e calcula a "Acurácia final" e a Matriz de Confusão prevendo **sobre os exatos mesmos pacientes que acabou de usar para treinar**.
* Compare nos seus próprios resultados da [Seção 2.1](file:///home/hector_furini/Projeto_ML_Python/projeto_hector/dev/script/projeto_hector/04_classificação.ipynb#L439-L1103):
  * **Melhor acurácia (CV)** *(validação cruzada real entre folds)*: fica entre **0.52 e 0.60** (quase um cara ou coroa).
  * **Acurácia final** *(avaliada no próprio conjunto)*: salta para **1.0000**.

---

### 2. Poucos pacientes para muitas combinações genéticas (131 linhas × 34 colunas)
Com 12 genes em *One-Hot Encoding* (~34 variáveis binárias), existem $3^{12} = 531.441$ combinações possíveis de genótipos, mas apenas **131 pacientes** no treino (e 44 na validação/teste). 

Quase todo paciente tem uma "assinatura genética" única na tabela. Para um modelo flexível, é muito fácil criar regras específicas que separam perfeitamente esses 131 indivíduos sem encontrar um padrão real da doença.

---

### 3. Os hiperparâmetros escolhidos permitem "decorar" os dados
Veja os hiperparâmetros que venceram no `GridSearchCV` do treino em [04_classificação.ipynb](file:///home/hector_furini/Projeto_ML_Python/projeto_hector/dev/script/projeto_hector/04_classificação.ipynb#L439-L1069):

| Modelo | Hiperparâmetro escolhido | Por que dá 100% no mesmo conjunto? |
| :--- | :--- | :--- |
| **k-NN** | `n_neighbors: 1` | Com $k=1$, o vizinho mais próximo de um paciente (quando você prevê nos mesmos dados de treino) **é ele mesmo** (distância zero). A acurácia será matematicamente 100% sempre que não houver dois pacientes idênticos com classes opostas. |
| **SVM** | `kernel: 'rbf'`, `C: 100` | Um `C=100` aplica penalidade altíssima para qualquer erro no treino, forçando a fronteira não-linear (`rbf`) a contornar cada paciente individualmente. |
| **Random Forest** | `max_depth: 10` (ou `None`), `min_samples_split: 2` | Com 131 amostras, uma árvore de profundidade 10 consegue criar até $2^{10} = 1024$ folhas — mais do que suficiente para isolar cada paciente em uma folha própria. |
| **GBM** | `max_depth: 5`, `n_estimators: 50`, `learning_rate: 0.2` | 50 árvores sequenciais de profundidade 5 corrigindo os resíduos de apenas 131 pontos zeram o erro de treino facilmente. |
| **Rede Neural (NN)** | `solver: 'lbfgs'`, `alpha: 0.0001`, 34 entradas × 10 neurônios | São ~360 pesos ajustáveis para apenas 131 amostras, com regularização mínima (`alpha` muito baixo) e otimizador quasi-Newton (`lbfgs`), convergindo para erro zero no treino. |
| **Naive Bayes (NB)** | *(Único que deu ~0.52)* | Como o Naive Bayes assume independência estrita entre os genes e só calcula médias e variâncias por classe, ele **não tem capacidade matemática** de memorizar combinações individuais. |

---

### A prova está na Seção 4 (Avaliação Holdout)
Na [tabela da Seção 4](file:///home/hector_furini/Projeto_ML_Python/projeto_hector/dev/script/projeto_hector/04_classificação.ipynb#L3719-L3748), onde os modelos foram treinados **apenas no Treino** e testados em pacientes inéditos (**Validação** e **Teste**), você vê o comportamento real:

* **No Treino:** k-NN, SVM, RF, GBM e NN têm Acurácia = `1.0000` e AUC = `1.0000`.
* **Na Validação e no Teste:** a Acurácia de todos cai para a faixa de **0.45 a 0.59**, e a AUC fica entre **0.42 e 0.64**.

Isso é o retrato clássico de **sobreajuste (*overfitting*)**: como os polimorfismos analisados têm baixa correlação direta com a recidiva, o `GridSearchCV` acaba selecionando `feature_selection__k: 'all'` (todas as 34 colunas) com hiperparâmetros complexos que memorizam os pacientes de treino, mas têm desempenho próximo ao acaso (50%) em novos pacientes.