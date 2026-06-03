import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import text
import sys
import os
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.connection import engine
from config.settings import INDEX_BASE_VALUE

logger = logging.getLogger(__name__)

def plot_performance(perf_df: pd.DataFrame, output_path: str = "assets/index_performance.png"):
    """
    Generates a professional performance chart comparing the custom index vs the benchmark.
    """
    if perf_df.empty:
        logger.warning("Performance dataframe is empty. No chart generated.")
        return
        
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    logger.info(f"Generating performance visualization at {output_path}...")
    
    # Set premium aesthetics
    sns.set_theme(style="whitegrid", palette="muted")
    plt.figure(figsize=(12, 6))
    
    # Ensure date column is datetime for plotting
    perf_df['date'] = pd.to_datetime(perf_df['date'])
    
    # Plot Index and Benchmark
    sns.lineplot(data=perf_df, x='date', y='index_value', label='IndexForge Custom', linewidth=2, color="#0072B2")
    sns.lineplot(data=perf_df, x='date', y='bench_rebased', label='SPY Benchmark (Rebased)', linewidth=1.5, linestyle='--', color="#D55E00")
    
    # Final touches
    plt.title("Index Performance Attribution: IndexForge vs. Benchmark", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel(f"Value (Base={INDEX_BASE_VALUE})", fontsize=12)
    plt.legend(frameon=True, fontsize=11)
    
    # Add a watermark or signature for professionalism
    plt.text(perf_df['date'].max(), perf_df['index_value'].min(), "IndexForge v2 Engineering Prototype", 
             fontsize=10, color='gray', alpha=0.5, ha='right', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    logger.info("Visualization saved successfully.")
