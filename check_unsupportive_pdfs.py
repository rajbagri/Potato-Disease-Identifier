"""
Diagnostic Script: Check Unsupportive PDFs
Analyzes PDFs in data/Unsupportive_data to identify issues
"""

import os
import fitz  # PyMuPDF

UNSUPPORTIVE_DIR = "data\Fertilization_data"

def analyze_pdf(pdf_path):
    """Analyze a PDF to identify issues"""
    
    pdf_name = os.path.basename(pdf_path)
    doc = fitz.open(pdf_path)
    
    issues = []
    page_details = []
    total_text_chars = 0
    total_images = 0
    full_page_images = 0
    scanned_pages = 0
    
    for page_num, page in enumerate(doc):
        # Extract text
        page_text = page.get_text()
        text_length = len(page_text.strip())
        total_text_chars += text_length
        
        # Get page dimensions
        page_rect = page.rect
        page_width = page_rect.width
        page_height = page_rect.height
        page_area = page_width * page_height
        
        # Check for scanned page (minimal text)
        is_scanned = text_length < 50
        if is_scanned:
            scanned_pages += 1
        
        # Analyze images on this page
        image_list = page.get_images(full=True)
        page_images = len(image_list)
        total_images += page_images
        
        page_has_fullpage_img = False
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            
            # Get image position and size
            img_rects = page.get_image_rects(xref)
            
            if img_rects:
                img_rect = img_rects[0]
                img_area = abs((img_rect.x1 - img_rect.x0) * (img_rect.y1 - img_rect.y0))
                coverage = img_area / page_area if page_area > 0 else 0
                
                # Check if full-page image
                if coverage > 0.85:
                    page_has_fullpage_img = True
                    full_page_images += 1
        
        # Record page details
        page_info = {
            'page_num': page_num + 1,
            'text_chars': text_length,
            'is_scanned': is_scanned,
            'num_images': page_images,
            'has_fullpage_image': page_has_fullpage_img
        }
        page_details.append(page_info)
    
    # Save total pages before closing
    total_pages = len(doc)
    doc.close()
    
    # Identify issues
    if scanned_pages > 0:
        issues.append(f"{scanned_pages} scanned page(s) with minimal text")
    
    if full_page_images > 0:
        issues.append(f"{full_page_images} full-page image(s) detected")
    
    if total_text_chars < 500:
        issues.append(f"Very low text content ({total_text_chars} chars total)")
    
    return {
        'filename': pdf_name,
        'total_pages': total_pages,
        'total_text_chars': total_text_chars,
        'total_images': total_images,
        'scanned_pages': scanned_pages,
        'full_page_images': full_page_images,
        'issues': issues,
        'page_details': page_details
    }


def main():
    print("\n" + "="*80)
    print("UNSUPPORTIVE PDF DIAGNOSTIC REPORT")
    print("="*80)
    
    if not os.path.exists(UNSUPPORTIVE_DIR):
        print(f"\n❌ Directory not found: {UNSUPPORTIVE_DIR}")
        print("   Please create this folder and add problematic PDFs")
        return
    
    # Get all PDFs in unsupportive folder
    pdf_files = [f for f in os.listdir(UNSUPPORTIVE_DIR) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"\n✓ No PDF files found in {UNSUPPORTIVE_DIR}")
        print("  All PDFs are supported!")
        return
    
    print(f"\nFound {len(pdf_files)} PDF(s) in {UNSUPPORTIVE_DIR}\n")
    
    all_results = []
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(UNSUPPORTIVE_DIR, pdf_file)
        print(f"Analyzing: {pdf_file}...")
        
        try:
            result = analyze_pdf(pdf_path)
            all_results.append(result)
            
            print(f"  Pages: {result['total_pages']}")
            print(f"  Text: {result['total_text_chars']} characters")
            print(f"  Images: {result['total_images']} total")
            print(f"  Scanned pages: {result['scanned_pages']}")
            print(f"  Full-page images: {result['full_page_images']}")
            
            if result['issues']:
                print(f"  ⚠️  Issues found:")
                for issue in result['issues']:
                    print(f"     - {issue}")
            else:
                print(f"  ✓ No issues detected")
            
            print()
            
        except Exception as e:
            print(f"  ❌ Error analyzing {pdf_file}: {e}\n")
    
    # Summary report
    print("="*80)
    print("SUMMARY")
    print("="*80)
    
    total_scanned = sum(r['scanned_pages'] for r in all_results)
    total_fullpage = sum(r['full_page_images'] for r in all_results)
    total_low_text = sum(1 for r in all_results if r['total_text_chars'] < 500)
    
    print(f"\nTotal PDFs analyzed: {len(all_results)}")
    print(f"Total scanned pages: {total_scanned}")
    print(f"Total full-page images: {total_fullpage}")
    print(f"PDFs with very low text: {total_low_text}")
    
    print("\n" + "="*80)
    print("DETAILED PAGE BREAKDOWN")
    print("="*80)
    
    for result in all_results:
        if result['scanned_pages'] > 0 or result['full_page_images'] > 0:
            print(f"\n📄 {result['filename']}")
            print(f"   {'Page':<6} {'Text':<12} {'Images':<8} {'Status'}")
            print(f"   {'-'*6} {'-'*12} {'-'*8} {'-'*30}")
            
            for page in result['page_details']:
                status_flags = []
                if page['is_scanned']:
                    status_flags.append("SCANNED")
                if page['has_fullpage_image']:
                    status_flags.append("FULL-PAGE IMAGE")
                
                status = ", ".join(status_flags) if status_flags else "OK"
                
                print(f"   {page['page_num']:<6} {page['text_chars']:<12} {page['num_images']:<8} {status}")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if total_scanned > 0 or total_fullpage > 0:
        print("\n🔧 Issues Detected - Solutions:")
        print("\n1. OCR Integration (Automated):")
        print("   - Install: pip install pytesseract pillow")
        print("   - Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki")
        print("   - Modify ingestion.py to add OCR support")
        
        print("\n2. Manual Pre-Processing (Quick Fix):")
        print("   - Upload PDFs to: https://www.ilovepdf.com/ocr_pdf")
        print("   - Download searchable PDFs")
        print("   - Move back to data/ folder")
        
        print("\n3. Adobe Acrobat (Highest Quality):")
        print("   - Open PDF → Tools → Recognize Text → In This File")
        print("   - Save as searchable PDF")
    else:
        print("\n✓ All PDFs look good! No major issues detected.")
    
    print("\n")


if __name__ == "__main__":
    main()