# Guia de Implementação e Prompting: Qwen-0.6B

Este documento resume a lógica de funcionamento e a estrutura necessária para operar o modelo Qwen-0.6B através do SDK de inferência local.

---

## 1. O Algoritmo de Geração (Ciclo de Baixo Nível)
A geração de texto é feita de forma incremental (Token a Token) através de métodos públicos:

1. **Encode (Fora do ciclo):** Transforma o prompt inicial composto num tensor/lista de IDs.
2. **Logits:** O método `get_logits_from_input_ids` recebe a lista acumulada e calcula as pontuações para todo o vocabulário.
3. **Seleção:** Encontra-se o índice (ID) do maior valor na lista de logits (`max(logits)`).
4. **Acumulação:** Adiciona-se o ID vencedor à lista de IDs anterior.
5. **Repetição:** O ciclo repete-se enviando a lista atualizada de volta para o passo 2.
6. **Decode (No final):** Transforma a lista final de IDs acumulados em texto legível.

---

## 2. Estrutura do Prompt (Separação e Controlo)
O Qwen utiliza a arquitetura ChatML. Para garantir o controlo absoluto do formato e separar as diretrizes da pergunta, use as tags nativas obrigatórias:

* `<|im_start|>system` -> Define as regras rígidas (ex: "Responda apenas em JSON") e injeta dados de ficheiros locais.
* `<|im_start|>user` -> Insere a pergunta ou dados do utilizador.
* `<|im_start|>assistant` -> Delimita as respostas do modelo.
* `<|im_end|>` -> Fecha cada bloco de mensagem.

**Template do Prompt a enviar ao Encode:**
```text
<|im_start|>system
[Diretrizes de formato e conteúdo do ficheiro]
<|im_end|>
<|im_start|>user
[Pergunta do utilizador]
<|im_end|>
<|im_start|>assistant
```
*Nota: Terminar o prompt exatamente em `<|im_start|>assistant` força o modelo a gerar apenas a resposta.*

---

## 3. Controlo de Fluxo e Comportamento de Tokens

### Como saber que o LLM acabou a resposta?
O modelo sinaliza o fim da resposta gerando o token especial **`<|im_end|>`**. No seu ciclo de iteração, assim que o índice do maior logit corresponder ao ID deste token, deve executar um `break` para interromper a geração imediatamente.

### O modelo pode repetir tokens?
**Sim.** Modelos ultrapequenos como o Qwen-0.6B entram facilmente em loops de repetição infinita se usar a busca puramente determinística (escolher sempre o maior logit absoluto). Para evitar que ele repita blocos de texto, aplica-se amostragem probabilística (sampling/temperatura) em vez de pegar apenas no valor máximo.
