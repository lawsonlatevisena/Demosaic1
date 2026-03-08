TEX=article.tex
NAME=$(basename $(TEX))
BUILD=build

.PHONY: odt docx clean ensure-build odt_word docx_word

ensure-build:
	mkdir -p $(BUILD)

odt: ensure-build
	(make4ht -f odt $(NAME).tex "mathml,charset=utf-8,pic-m" -d $(BUILD)) || (mk4ht oolatex $(NAME).tex && mv -f $(NAME).odt $(BUILD)/$(NAME).odt)

docx: odt
	libreoffice --headless --convert-to docx --outdir $(BUILD) $(BUILD)/$(NAME).odt

# Word-safe conversion using the copy article_word.tex
WORDTEX=article_word.tex
WORDNAME=$(basename $(WORDTEX))

odt_word: ensure-build
	(make4ht -f odt $(WORDNAME).tex "mathml,charset=utf-8,pic-m" -d $(BUILD)) || (mk4ht oolatex $(WORDNAME).tex && mv -f $(WORDNAME).odt $(BUILD)/$(WORDNAME).odt)

docx_word: odt_word
	libreoffice --headless --convert-to docx --outdir $(BUILD) $(BUILD)/$(WORDNAME).odt

clean:
	rm -rf $(BUILD)
