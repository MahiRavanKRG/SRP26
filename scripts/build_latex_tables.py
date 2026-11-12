import os
import re

os.makedirs('paper/tables', exist_ok=True)

def parse_val(v):
    cleaned = re.sub(r'[\*\_\<\>u\/]', '', v).strip()
    if cleaned in ['–', '-', 'n/r', '', '\u2013', '_n/r_']:
        return None
    try:
        return float(cleaned)
    except:
        return None

def clean_txt(v):
    return re.sub(r'[\*\_\<\>u\/]', '', v).strip()

def generate_table(md_path, out_tex_path, is_nodep=False):
    with open(md_path, encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip().startswith('|')][3:]
    
    datasets = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'] if is_nodep else ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Traffic', 'Electricity', 'Solar']
    horizons = ['96', '192', '336', '720', 'Avg']
    
    out_lines = []
    
    for d_idx, ds in enumerate(datasets):
        for h_idx, hz in enumerate(horizons):
            line_idx = d_idx * 5 + h_idx
            parts = [p.strip() for p in lines[line_idx].split('|')[1:-1]]
            
            # parse numeric values for ranking
            mses = [parse_val(parts[1 + i*2]) for i in range(8)]
            maes = [parse_val(parts[2 + i*2]) for i in range(8)]
            
            valid_mses = sorted(list(set([m for m in mses if m is not None])))
            valid_maes = sorted(list(set([a for a in maes if a is not None])))
            
            best_m = valid_mses[0] if valid_mses else None
            second_m = valid_mses[1] if len(valid_mses) > 1 else None
            
            best_a = valid_maes[0] if valid_maes else None
            second_a = valid_maes[1] if len(valid_maes) > 1 else None
            
            row_cells = []
            for col in range(8):
                m_raw = parts[1 + col*2].strip()
                a_raw = parts[2 + col*2].strip()
                m_val = mses[col]
                a_val = maes[col]
                
                # format MSE
                if m_val is None:
                    m_str = '--'
                elif m_val == best_m or ('**' in m_raw):
                    m_str = '\\textbf{' + clean_txt(m_raw) + '}'
                elif m_val == second_m:
                    m_str = '\\underline{' + clean_txt(m_raw) + '}'
                else:
                    m_str = clean_txt(m_raw)
                    
                # format MAE
                if a_val is None:
                    a_str = '--'
                elif a_val == best_a or ('**' in a_raw):
                    a_str = '\\textbf{' + clean_txt(a_raw) + '}'
                elif a_val == second_a:
                    a_str = '\\underline{' + clean_txt(a_raw) + '}'
                else:
                    a_str = clean_txt(a_raw)
                    
                row_cells.extend([m_str, a_str])
                
            ds_label = '\\textbf{' + ds + '}' if h_idx == 0 else ''
            hz_label = hz if hz != 'Avg' else '\\textit{Avg}'
            
            formatted_cells = ' & '.join(f'{c:>14}' for c in row_cells)
            out_lines.append(f'{ds_label:14} & {hz_label:10} & {formatted_cells} \\\\')
            
        if d_idx != len(datasets) - 1:
            out_lines.append('\\midrule')
            
    header = r'''\begin{tabular}{llcccccccccccccccc}
\toprule
\textbf{Dataset} & \textbf{Horizon} & \multicolumn{2}{c}{\textbf{Linear (ours)}} & \multicolumn{2}{c}{\textbf{MLP (ours)}} & \multicolumn{2}{c}{\textbf{CNN (ours)}} & \multicolumn{2}{c}{\textbf{SOR (ours)}} & \multicolumn{2}{c}{\textbf{CycleNet/L}} & \multicolumn{2}{c}{\textbf{CycleNet/M}} & \multicolumn{2}{c}{\textbf{PatchTST/42}} & \multicolumn{2}{c}{\textbf{HDMixer}} \\
\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8} \cmidrule(lr){9-10} \cmidrule(lr){11-12} \cmidrule(lr){13-14} \cmidrule(lr){15-16} \cmidrule(lr){17-18}
& & MSE & MAE & MSE & MAE & MSE & MAE & MSE & MAE & MSE & MAE & MSE & MAE & MSE & MAE & MSE & MAE \\
\midrule
'''
    full_table = header + '\n'.join(out_lines) + '\n\\bottomrule\n\\end{tabular}\n'
    with open(out_tex_path, 'w', encoding='utf-8') as f:
        f.write(full_table)
    print(f'Wrote {out_tex_path}')

if __name__ == '__main__':
    t1 = 'docs/Table.md' if os.path.exists('docs/Table.md') else 'Table.md'
    t2 = 'docs/TableNoDependence.md' if os.path.exists('docs/TableNoDependence.md') else 'TableNoDependence.md'
    generate_table(t1, 'paper/tables/table_main_benchmark.tex', is_nodep=False)
    generate_table(t2, 'paper/tables/table_channel_independence.tex', is_nodep=True)

