from app.api.pdf_processing import DocumentProcessor
import re
from pathlib import Path
from unstructured.partition.pdf import partition_pdf
import spacy
from langchain.text_splitter import SpacyTextSplitter

PDF_DIR = Path(__file__).parent / "data"

nlp = spacy.load("en_core_web_sm")


def test_filter_docs():
    text = """better connected events. But the scatter was quite large even for events with small longitudinal separations from the footpoint of the Parker spiral.
In this paper, we further investigate two events from Kihara et al. (2020) that apparently had different TO, despite their similar source locations in the western hemisphere and similar CME speeds of ∼1200 km s−1. We explore the possibility that the event with longer TO may reflect a slow growth of the CME-driven shock wave that becomes strong enough for particle acceleration only at later times. Combining CME height-time profiles with radio dynamic spectra that contain type II radio bursts, we follow the temporal evolution of the Alfv ́en Mach number of the shock wave with time above the two active regions without conducting advanced modeling. In Section 2, we describe the event selection and give an overview of the two events. We revisit in Section 3 the SEP timescales that are used for the subsequent analysis. In Section 4, we study how the shock waves develop in the two events in relation to TO or the SEP release times. In addition we study other factors that may affect these times. We summarize our findings in Section 5.
2. OBSERVATIONS 2.1. Event Selection
Kihara et al. (2020) conducted a statistical study of energetic CMEs that occurred between December 2006 and October 2017 in terms of their associations with SEP events. They also studied the timescales of the associated SEP events with respect to the speeds and source locations of the CMEs as shown in the Table 2 of Kihara et al. (2020). They based the SEP analysis on data from the Energetic Particle Sensor (Onsager et al. 1996) on the Geostationary Operations Environmental Satellite (GOES), and the High-Energy Telescope (HET; von Rosenvinge et al. 2008) and the Low-Energy Telescope (LET; Mewaldt et al. 2008), which belong to the suite of instruments for the In Situ Measurements of Particles and CME Transients (IMPACT; Luhmann et al. 2008) on the Solar-Terrestrial Relations Observatory (STEREO; Kaiser et al. 2008). The SEP events were identified when the >10 MeV proton flux exceeded 1 particle flux unit (pfu; defined as particles s−1 sr−1 cm−2). The CMEs responsible for the SEP events and the associated flares were found in white-light coronagraph and EUV low-coronal images produced by the instruments on the Solar and Heliospheric Observatory (SOHO; Domingo et al. 1995), Solar Dynamics Observatory (SDO; Pesnell et al. 2012), and STEREO. As expected, Kihara et al. (2020) found that SEP events that occurred in regions not far from the magnetic footpoints of the observer tend to have shorter timescales (in both TO and TR, see Figure 6 of the paper). However, TO mostly (77/82) ranges from 0.5 to 4 hours even when the longitudinal separation of the region from the Parker spiral footpoint is less than 60◦. TO also appears to depend on the speed of the associated CME.
In this paper, we selected two events that have widely different TO (i.e., 62 and 158 minutes) even though they came from regions in similar longitudes and were associated with halo CMEs with similar speeds. They occurred on 2014 April 18 and 2017 July 14. We hereafter refer to these SEP events as Event 1 and Event 2, respectively. Their basic parameters are shown in Table 1. The primary purpose of this work is to explain this wide difference in TO. We also revise the SEP onset times in Section 3, which we will use in the subsequent analyses.
2.2. Overview of the Events
In Figure 1 we plot the soft X-ray (SXR) and SEP (proton) time profiles of the two events over two-day intervals. The flare associated with Event 1 (in panel (a)) is M7.3 in the GOES classification (the peak 1 – 8  ̊A flux of 7.3×10−5 W m−2), whereas the one associated with Event 2 (in panel (c)) is M2.4. The latter flare is of much longer duration, staying above the pre-event level in the GOES 1–8  ̊A channel for more than two days. Both flares are associated with halo CMEs, whose mean linear speed is ∼1200 km s−1 across the combined field of view (FOV) of the C2 and C3 telescopes of the Large Angle Spectrometric Coronagraph (LASCO; Brueckner et al. 1995) on board SOHO. The CME launch times in black dashed lines are calculated by extrapolating the height-time relations from the LASCO C2 and C3 data to the unit height (1 solar radius R⊙), i.e. the solar surface.
Both Event 1 and Event 2 are accompanied by type II radio bursts, while their appearances are quite different as found in Figure 2, where we show radio dynamic spectra between 180 MHz and 0.1 MHz that consist of data from ground-based observatories and the Radio and Plasma Wave Experiment (WAVES; Bougeret et al. 1995) on the Wind spacecraft. In Event 1 (Figure 2(a)), the type II radio burst started at 12:55 UT from about 60 MHz (fundamental),"""
    text_splitter = SpacyTextSplitter(chunk_size=500, chunk_overlap=20)
    splits = text_splitter.split_text(text)
    print(splits)
    assert False


def is_junk(s):
    # Remove single and double quotations in string
    s = re.sub(r"['\"]", "", s)

    # Check for url-like patterns
    url_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    if re.search(url_pattern, s):
        return True

    # Check if the string has unusual spacing (a space after every character)
    unusual_spacing_pattern = re.compile(r"^((\w\s)+)$")
    if re.match(unusual_spacing_pattern, s):
        return True

    # Check if the string contains a high proportion of non-alphabetic characters
    non_alpha = len([c for c in s if not c.isalpha()])
    if non_alpha > len(s) / 2:
        return True

    # Check if the string matches a citation pattern
    citation_pattern = re.compile(r"(\w+[\.,]\s*)+([\da-zA-Z]{2,}\.?.*|$)")
    if re.match(citation_pattern, s.strip()):
        return True

    # If none of the above conditions were met, the string is not "junk"
    return False


def is_figure_caption(s):
    # Define the figure caption pattern
    figure_caption_pattern = re.compile(r"^Figure\s\d+\. .*$", re.DOTALL)

    # Check if the string matches the figure caption pattern
    return bool(re.match(figure_caption_pattern, s.strip()))


def test_unstructured():
    pdf_file = PDF_DIR / "soho_several_instruments.pdf"
    elements = partition_pdf(str(pdf_file), include_page_breaks=True)
    last_narrative_first_page = elements[12].text
    first_narrative_second_page = elements[20].text
    # ^ hopefully can get with narrative + junk test
    print(elements)


def test_get_rect_locations():
    pdf_file = PDF_DIR / "soho_several_instruments.pdf"
    assert pdf_file.exists()
    with DocumentProcessor(pdf_file) as doc_processor:
        documents = doc_processor.get_rect_locations()
    # documents = DocumentProcessor.filter_docs(documents)
    contents = [d.page_content for d in documents]
    content_str = "\n\n----------\n\n".join(contents)
    print(content_str)
    assert False


def test_get_location_dct():
    assert False
