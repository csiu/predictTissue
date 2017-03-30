PROJDIR := /projects/csiu_prj_results/PROJECTS/predictTissue/

# Aux files
finder_list := $(PROJDIR)metadata/listoffiles.h3k4me3.finder.txt
finder_dir := $(PROJDIR)data/finder/h3k4me3/

bin_file := $(PROJDIR)reference/hg19.200bpbins.bed.srt

# Paths
feature_dir := $(PROJDIR)results/features/promoter/_features/
ef_dir := $(feature_dir)extractFeatures_samples/

tss_bed := $(feature_dir)HS.GRCh37.75.gtf.chr.pcgene.tss.bed
features_bed := $(feature_dir)features.bed

# Features -------------------------------------------------------------------
# Produces feature input file for training
# CSV Format: sample_1,sample_2,sample_3,...

feature_matrix := $(PROJDIR)results/features/promoter/input.txt
all: $(feature_matrix)

# Ensure enrichment files are there, a hack (FIXME)
$(finder_dir)%.bed.gz: $(finder_list)
	perl -pe 's%^(.*)(CEMT_\d+)(.*)$$%[[ -L $(finder_dir)$$2.bed.gz ]] || ln -s $$1$$2$$3 $(finder_dir)$$2.bed.gz%' $< | sh

# Create feature file
$(features_bed): $(bin_file) $(tss_bed)
	bedtools window -a $(tss_bed) -b $(bin_file) -w 1000 \
		| cut -f4-6 | sort | uniq | bedtools sort -i - \
		> $@

# Extract features
$(ef_dir)%.ef: $(finder_dir)%.bed.gz $(features_bed)
	bedtools intersect -a $(features_bed) -b $< -c \
		| awk -F'\t' -v sample_id='$*' 'BEGIN{print sample_id} {print $$NF}' \
		> $@

# Join
CEMT_samples := $(shell perl -pe 's/^.*(CEMT_\d+).*/$$1/' $(finder_list))
$(feature_matrix): $(addprefix $(ef_dir), $(addsuffix .ef, $(CEMT_samples)))
	paste -d',' $^ \
	| sed '2,$$s|2|1|g' > $@
