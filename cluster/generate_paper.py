from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER

def generate_paper():
    doc = SimpleDocTemplate("Aaditya_Chaturvedi_MTS_Research.pdf", pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], alignment=TA_CENTER, fontSize=16, spaceAfter=12)
    author_style = ParagraphStyle('Author', parent=styles['Normal'], alignment=TA_CENTER, fontSize=11, spaceAfter=20)
    heading_style = ParagraphStyle('Heading', parent=styles['Heading2'], fontSize=12, spaceBefore=12, spaceAfter=6)
    body_style = ParagraphStyle('Body', parent=styles['Normal'], alignment=TA_JUSTIFY, fontSize=10, leading=12)
    
    content = []
    
    # Title & Author
    content.append(Paragraph("Spectral Integrity in Lightweight Multivariate Time Series Forecasting: An Empirical Study on Decomposition-based Linear Models", title_style))
    content.append(Paragraph("Aaditya Chaturvedi", author_style))
    content.append(Paragraph("University of Hildesheim | International Masters in Data Analytics", author_style))
    content.append(Spacer(1, 12))
    
    # Abstract
    content.append(Paragraph("Abstract", heading_style))
    content.append(Paragraph("This paper investigates the efficacy of decomposition-based linear models for Multivariate Time Series (MTS) forecasting under strict computational constraints. Utilizing the DLinear architecture, we evaluate performance on the ETTh1 benchmark dataset. Beyond standard error metrics, we introduce frequency-domain diagnostics, including band-wise error decomposition and Log-Spectral Distance (LSD), to quantify the model's ability to preserve seasonal and trend components. Our results demonstrate that while linear models effectively capture high-frequency noise, preserving low-frequency trend integrity remains a critical challenge for edge-deployed architectures.", body_style))
    
    # 1. Introduction
    content.append(Paragraph("1. Introduction", heading_style))
    content.append(Paragraph("Multivariate Time Series forecasting is a cornerstone of data analytics, with applications ranging from energy grid management to financial modeling. While Transformer-based models have dominated the SOTA, their computational overhead often precludes deployment on edge hardware. This study focuses on the 'Extreme Lightweight' pathway, utilizing decomposition-based linear models to achieve high-fidelity predictions with minimal VRAM usage.", body_style))
    
    # 2. Methodology
    content.append(Paragraph("2. Methodology", heading_style))
    content.append(Paragraph("We employ the DLinear architecture, which decomposes the input signal into trend and seasonal components using moving average kernels. The trend component captures long-term shifts, while the seasonal component focuses on periodic fluctuations. Predictions are generated via independent linear layers for each component and recombined to form the final forecast.", body_style))
    
    # 3. Experimental Setup
    content.append(Paragraph("3. Experimental Setup", heading_style))
    content.append(Paragraph("The experiment was conducted on a local machine equipped with an NVIDIA RTX 4050 GPU (6GB VRAM). The ETTh1 dataset was utilized, consisting of 17,420 observations across 7 features. We utilized a sequence length of 96 and a prediction horizon of 24. Optimization was performed using AdamW with gradient accumulation to maximize hardware efficiency.", body_style))
    
    # 4. Results
    content.append(Paragraph("4. Results", heading_style))
    content.append(Paragraph("The model achieved convergence within 5 epochs. Quantitative analysis was performed using a custom frequency-domain suite to identify specific failure modes.", body_style))
    
    data = [
        ['Metric', 'Value', 'Interpretation'],
        ['MSE_LOW', '0.264206', 'Error in Trend/Low-Freq'],
        ['MSE_MID', '0.054879', 'Error in Intermediate-Freq'],
        ['MSE_HIGH', '0.008612', 'Error in Noise/High-Freq'],
        ['LSD (Dist)', '10.548798', 'Spectral Distortion'],
    ]
    
    t = Table(data, colWidths=[100, 100, 150])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    content.append(t)
    content.append(Spacer(1, 12))
    content.append(Paragraph("The results indicate that the model is exceptionally proficient at capturing high-frequency components (MSE_HIGH: 0.0086), suggesting a strong resistance to over-smoothing. However, the significantly higher MSE_LOW (0.2642) highlights the difficulty of predicting long-term trend shifts in non-stationary energy data.", body_style))
    
    # 5. Conclusion
    content.append(Paragraph("5. Conclusion", heading_style))
    content.append(Paragraph("Our findings validate the use of lightweight linear models for edge-based MTS forecasting. The introduction of band-wise error decomposition provides a more granular understanding of model performance than traditional MSE. Future work will explore the integration of Stationary Wavelet Transforms (SWT) to further isolate seasonal components and improve low-frequency trend accuracy.", body_style))
    
    doc.build(content)
    print("PDF Generated: Aaditya_Chaturvedi_MTS_Research.pdf")

if __name__ == "__main__":
    generate_paper()
