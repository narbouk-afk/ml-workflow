
def displayResult(precisions: list[float], recalls: list[float], precision_test: float, recall_test: float):
    fold_str = " ".join([f'fold{i+1}' for i in range(len(precisions))])
    precisions_str = "  ".join([f'{prec:.2f}' for prec in precisions])
    recalls_str = "  ".join([f'{recal:.2f}' for recal in recalls])
    print(f'          {fold_str}')
    print(f'Precision  {precisions_str}')
    print(f'Recall     {recalls_str}')
    print(f'Test precision : {precision_test:.3f}%')
    print(f'Test recall    : {recall_test:.3f}%')
