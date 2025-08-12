import os
import re
import numpy as np
import pandas as pd
import pickle
from ms_deisotope import MzMLLoader
import ms_deisotope


def get_data_mzml(xlsx_path, mzml_path, dat_path, log_str=[]):
    # This function is used for extracting data from mzML files and excel files
    # and write result into dat file using pickle.dump()
    # 读取单个mzML文件

    # Read excel file and drop some columns not used
    data = pd.DataFrame(pd.read_excel(xlsx_path, 'P'))
    drop_list = (['Matched_Reporter_Number', 'Matched_Reporter_Ions', 'GlycanScore', 'structure_coding', 'Corefucose',
                  'Bisection', 'Corefucose_decoy', 'Bisection_decoy', 'Glycan_decoy', 'Isoforms'])
    data = data.drop(columns=drop_list)
    # data = data.loc[data['FDR'] < 0.01]
    data.reset_index()

    # get mz/I from mzml file.  monoisotopic info
    run = MzMLLoader(mzml_path)
    monoisotopic_info = []
    isotopic_cluster = []
    mzi_info = []
    error_list = []
    item_n = data['LowEnergy_MS2Scan'].shape[0]
    finished_n = 0
    for i in range(item_n):
        j = data['LowEnergy_MS2Scan'].get(i)
        try:
            print(log_str + 'item' + str(finished_n) + '/' + str(item_n) + ':' + str(j) + '\n')
            scan = run[j - 1]
        except:
            error_list.append(j)
            mzi_info.append([])
            monoisotopic_info.append([])
            isotopic_cluster.append([])
        else:
            finished_n += 1
            peaks = scan.peaks
            mz = peaks.scan.arrays.mz
            mz = mz.reshape((len(mz), 1))
            inten = peaks.scan.arrays.intensity
            inten = inten.reshape((len(inten), 1))
            mzi_info.append(np.concatenate((mz, inten), axis=1))
            deconvoluted_peaks, _ = ms_deisotope.deconvolute_peaks(peaks, charge_range=(1, 4), priority_list=None,
                                                                   averagine=ms_deisotope.glycopeptide,
                                                                   truncate_after=0.8,
                                                                   deconvoluter_type=ms_deisotope.deconvolution.AveraginePeakDependenceGraphDeconvoluter,
                                                                   scorer=ms_deisotope.MSDeconVFitter(10.))
            deconvoluted_peaks = deconvoluted_peaks._mz_ordered
            monoisotopic = np.array([[0, 0, 0, 0]])
            isotopic = []
            for k in range(len(deconvoluted_peaks)):
                item = deconvoluted_peaks[k]
                pairs = item.envelope.pairs

                # monoisotopic = np.append(monoisotopic, [[item.mz, item.charge, item.intensity, pairs[0].intensity]])
                first_signal = 1
                monoisotopic_name = item.mz
                isotopic_list = []
                for m in range(len(pairs)):
                    if pairs[m].intensity != 1.0:
                        if first_signal == 1:
                            monoisotopic = np.append(monoisotopic,
                                                     [[pairs[m].mz, item.charge, item.intensity, pairs[m].intensity]])
                            monoisotopic_name = pairs[m].mz
                            first_signal = 0
                        isotopic_list.append([pairs[m].mz, pairs[m].intensity])
                isotopic.append({monoisotopic_name: isotopic_list})

            monoisotopic = monoisotopic.reshape((k + 2, 4), order='C')
            monoisotopic_info.append(np.delete(monoisotopic, 0, axis=0))
            isotopic_cluster.append(isotopic)

    print(error_list)
    print(len(error_list))
    data['mz_Intensity_Info'] = mzi_info
    data['MonoIsopicPeaks_Info'] = monoisotopic_info
    data['L_isotopicCluster'] = isotopic_cluster

    data = data.to_dict(orient='records')
    with open(dat_path, 'wb') as f:
        pickle.dump(data, f)


def get_data_batch_mzml(filepath_read, filepath_write):
    # 批量读取mzML文件
    path = filepath_read
    dirs = os.listdir(path)
    num = 0
    n_mzml = int(len(dirs) / 3)
    for item in dirs:
        fname = os.path.splitext(item)
        if fname[1] == '.mzML':
            num += 1
            # if fname[0] != 'Fut8_KO_hilic_brain_1':
            #     continue
            get_data_mzml(filepath_read + '\\' + fname[0] + '_result.xlsx',
                          filepath_read + '\\' + item,
                          filepath_write + '\\' + fname[0] + '_result.dat',
                          log_str='file:' + str(num) + '/ ' + str(n_mzml))


def extract_feature(monoisotopic_info, theor_mass, feature_name, ppm=20, score=False):
    # 提取特征值
    reslut = {}
    for i in range(len(theor_mass)):
        reslut.update(get_peak_matched(monoisotopic_info, theor_mass[i],
                                       feature_name[i], ppm, score=score))
    return reslut


def extract_feature_batch(filepath_read, filepath_write, ppm=20, score=False):
    # 批量提取特征值
    dirs = os.listdir(filepath_read)
    num = 0
    if not score:
        feature_db = np.array([203.0794, 406.1588, 568.2116, 730.2644, 892.3172, 771.291,
                               349.1373, 552.2167, 714.2695, 876.3223, 1038.3751, 917.3489])
        feature_name = ['N1', 'N2', 'N2H1', 'N2H2', 'N2H3', 'N3H1',
                        'N1F1', 'N2F1', 'N2H1F1', 'N2H2F1', 'N2H3F1', 'N3H1F1']
        new_key = 'Matched_Feature_ions_' + str(ppm) + 'ppm'
    else:
        feature_db_target = np.array([609.2382, 771.291, 917.3489])
        feature_name_target = ['N3', 'N3H1', 'N3H1F1']
        new_key_target = 'Matched_Feature_ions_' + str(ppm) + 'ppm_target'
        feature_name_decoy = ['N3H1', 'N3H1F1']
        new_key_decoy = 'Matched_Feature_ions_' + str(ppm) + 'ppm_decoy'
    n_dat = len(dirs)
    for item in dirs:
        file_name = os.path.splitext(item)
        if file_name[1] == '.dat':
            with open(filepath_read + '\\' + item, 'rb') as f:
                data = pickle.load(f)
            for i in range(len(data)):
                if len(data[i]['MonoIsopicPeaks_Info']):
                    monoisotopic_info = data[i]['MonoIsopicPeaks_Info']
                    pep_mass = data[i]['PeptideMass']
                    if not score:
                        theor_mass = feature_db + pep_mass
                        data[i][new_key] = extract_feature(monoisotopic_info, theor_mass,
                                                           feature_name=feature_name, ppm=ppm, score=score)
                    else:
                        n3h1_decoy = 771.291 + 5 + 34 * np.random.random()
                        n3h1f1_decoy = 917.3489 + 5 + 44 * np.random.random()
                        while (933.3438 - 5) < n3h1f1_decoy < (933.3438 + 5):
                            n3h1f1_decoy = 917.3489 + 5 + 44 * np.random.random()
                        feature_db_decoy = np.array([n3h1_decoy, n3h1f1_decoy])
                        theor_mass = feature_db_target + pep_mass
                        data[i][new_key_target] = extract_feature(monoisotopic_info, theor_mass,
                                                                  feature_name=feature_name_target, ppm=ppm, score=score)
                        theor_mass = feature_db_decoy + pep_mass
                        data[i][new_key_decoy] = extract_feature(monoisotopic_info, theor_mass,
                                                                 feature_name=feature_name_decoy, ppm=ppm, score=score)
            with open(filepath_write + '\\' + item, 'wb') as f:
                pickle.dump(data, f)
                num += 1
                print('Feature extraction:' + str(num) + '/' + str(n_dat))


def get_peak_matched(monoisotopic_info, theor_mass, peak_str, ppm=20, score=False):
    mass = monoisotopic_info[:, 0] * monoisotopic_info[:, 1] - 1.00727646677 * monoisotopic_info[:, 1]
    inten = monoisotopic_info[0:, 2]
    inten_monoisotopic = 0
    merr = abs(mass - theor_mass)
    index = np.where(merr < ppm * (theor_mass) / 1000000)
    merr_ratio = 1
    if len(index[0]):
        inten_monoisotopic = max(inten[index])
        index_max = np.argmax(inten[index])
        if isinstance(index_max, list):
            index_max = index[0][index_max[0]]
        else:
            index_max = index[0][index_max]
        merr_ratio = merr[index_max] / (ppm * (theor_mass) / 1000000)
    # +1
    merr = abs(mass - (theor_mass + 1.003354835))
    index = np.where(merr < ppm * (theor_mass + 1.003354835) / 1000000)
    if len(index[0]):
        tmp = max(inten[index])
        if inten_monoisotopic < tmp:
            inten_monoisotopic = tmp
            index_max = np.argmax(inten[index])
            if isinstance(index_max, list):
                index_max = index[0][index_max[0]]
            else:
                index_max = index[0][index_max]
            merr_ratio = merr[index_max] / (ppm * (theor_mass + 1.003354835) / 1000000)
    # -1
    merr = abs(mass - (theor_mass - 1.003354835))
    index = np.where(merr < ppm * (theor_mass - 1.003354835) / 1000000)
    if len(index[0]):
        tmp = max(inten[index])
        if inten_monoisotopic < tmp:
            inten_monoisotopic = tmp
            index_max = np.argmax(inten[index])
            if isinstance(index_max, list):
                index_max = index[0][index_max[0]]
            else:
                index_max = index[0][index_max]
            merr_ratio = merr[index_max] / (ppm * (theor_mass - 1.003354835) / 1000000)
    # +2
    merr = abs(mass - (theor_mass + 2 * 1.003354835))
    index = np.where(merr < ppm * (theor_mass + 2 * 1.003354835) / 1000000)
    if len(index[0]):
        tmp = max(inten[index])
        if inten_monoisotopic < tmp:
            inten_monoisotopic = tmp
            index_max = np.argmax(inten[index])
            if isinstance(index_max, list):
                index_max = index[0][index_max[0]]
            else:
                index_max = index[0][index_max]
            merr_ratio = merr[index_max] / (ppm * (theor_mass + 2 * 1.003354835) / 1000000)
    # -2
    merr = abs(mass - (theor_mass - 2 * 1.003354835))
    index = np.where(merr < ppm * (theor_mass - 2 * 1.003354835) / 1000000)
    if len(index[0]):
        tmp = max(inten[index])
        if inten_monoisotopic < tmp:
            inten_monoisotopic = tmp
            index_max = np.argmax(inten[index])
            if isinstance(index_max, list):
                index_max = index[0][index_max[0]]
            else:
                index_max = index[0][index_max]
            merr_ratio = merr[index_max] / (ppm * (theor_mass - 2 * 1.003354835) / 1000000)

    if not score:
        if inten_monoisotopic:
            result = {peak_str: [monoisotopic_info[index_max, 0], monoisotopic_info[index_max, 1], inten_monoisotopic,
                                 theor_mass]}
        else:
            result = {peak_str: [0, 0, 0, 0]}
    else:
        if inten_monoisotopic:
            result = {peak_str: [monoisotopic_info[index_max, 0], monoisotopic_info[index_max, 1], inten_monoisotopic,
                                 theor_mass, merr_ratio]}
        else:
            result = {peak_str: [0, 0, 0, 0, 0]}
    return result


def get_peak_matched_Bion(monoisotopic_info, theor_mass, peak_str, ppm=20, score=False):
    mass = monoisotopic_info[:, 0] * monoisotopic_info[:, 1] - 1.00727646677 * monoisotopic_info[:, 1]
    inten = monoisotopic_info[0:, 2]
    inten_monoisotopic = 0
    merr = abs(mass - theor_mass)
    index = np.where(merr < ppm * (theor_mass) / 1000000)
    merr_ratio = 1
    if len(index[0]):
        inten_monoisotopic = max(inten[index])
        index_max = np.argmax(inten[index])
        if isinstance(index_max, list):
            index_max = index[0][index_max[0]]
        else:
            index_max = index[0][index_max]
        merr_ratio = merr[index_max] / (ppm * (theor_mass) / 1000000)

    if not score:
        if inten_monoisotopic:
            result = {peak_str: [monoisotopic_info[index_max, 0], monoisotopic_info[index_max, 1], inten_monoisotopic,
                                 theor_mass]}
        else:
            result = {peak_str: [0, 0, 0, 0]}
    else:
        if inten_monoisotopic:
            result = {peak_str: [monoisotopic_info[index_max, 0], monoisotopic_info[index_max, 1], inten_monoisotopic,
                                 theor_mass, merr_ratio]}
        else:
            result = {peak_str: [0, 0, 0, 0, 0]}
    return result


def getdata_core(file_path, feature_name, sample_type='MB', clf=None, file_name=None, ppm=20):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    x = np.zeros((1, len(feature_name)), dtype=float, order='C')
    result = []
    feature_value_key = 'Matched_Feature_ions_' + str(ppm) + 'ppm'
    for item in data:
        if len(item['MonoIsopicPeaks_Info']) == 0:
            continue
        info = dict(item[feature_value_key])
        feature_value = []
        for i in feature_name:
            feature_value.append(info[i][2])
        feature_value = np.array(feature_value)
        feature_value = feature_value.reshape((1, len(feature_value)), order='A')
        if sample_type == 'MB':
            if item['Glycan_type'] != 'High mannose':
                continue

        glycan_comp = item['GlycanComposition']
        tmp = re.findall(r'F', glycan_comp)
        if len(tmp):
            if clf is None:
                x = np.concatenate((x, feature_value), axis=0)
            else:
                label_predict = clf.predict(feature_value)
                post_prior = clf.predict_proba(feature_value)
                if label_predict:
                    item['Prediction_state'] = 'correct'
                else:
                    item['Prediction_state'] = 'incorrect'
                item['Core_Fucosylation_postprobability'] = post_prior[0, 1]
                item['File_name'] = file_name
                result.append(item)

    if clf is None:
        x = np.delete(x, 0, axis=0)
        return x
    else:
        return result


def getdata_all(file_path, feature_name, to_matlab=0, clf=None, file_name=None, ppm=20):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    x = np.zeros((1, len(feature_name)), dtype=float, order='C')
    result = []
    feature_value_key = 'Matched_Feature_ions_' + str(ppm) + 'ppm'
    for item in data:
        if len(item['MonoIsopicPeaks_Info']) == 0:
            continue
        info = dict(item[feature_value_key])
        feature_value = []
        for i in feature_name:
            feature_value.append(info[i][2])
        feature_value = np.array(feature_value)
        feature_value = feature_value.reshape((1, len(feature_value)), order='A')
        if to_matlab:
            # export outliers to matlab GUI
            idx = np.where(feature_value > 0)
            if idx[0].size:
                feature_value = feature_value / np.max(feature_value)
                if feature_value[0, 5] == 1 or feature_value[0, 6] == 1 or feature_value[0, 7] == 1 \
                        or feature_value[0, 8] == 1 or feature_value[0, 9] == 1:
                    item['File_name'] = file_name
                    result.append(item)
        else:
            if clf is None:
                x = np.concatenate((x, feature_value), axis=0)
            else:
                label_predict = clf.predict(feature_value)
                post_prior = clf.predict_proba(feature_value)
                if label_predict:
                    item['Prediction_state'] = 'incorrect'
                else:
                    item['Prediction_state'] = 'correct'
                item['Core_Fucosylation_postprobability'] = post_prior[0, 1]
                item['File_name'] = file_name
                result.append(item)

    if to_matlab:
        return result
    else:
        if clf is None:
            x = np.delete(x, 0, axis=0)
            return x
        else:
            return result


def getdata_core_uncore(file_path, feature_name, file_name, ppm=20):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    result_core = np.zeros((1, len(feature_name)), dtype=float, order='C')
    result_uncore = np.zeros((1, len(feature_name)), dtype=float, order='C')
    feature_value_key = 'Matched_Feature_ions_' + str(ppm) + 'ppm'
    data_exc = pd.DataFrame(pd.read_excel('E://python_project//GlycoPrediction//data//MB//' + file_name + '.xlsx', sheet_name='P'))
    item_n = data_exc['LowEnergy_MS2Scan'].shape[0]
    for i in range(item_n):
        item = data[i]
        if len(item['MonoIsopicPeaks_Info']) == 0:
            continue
        info = dict(item[feature_value_key])
        feature_value = []
        for j in feature_name:
            feature_value.append(info[j][2])
        feature_value = np.array(feature_value)
        feature_value = feature_value.reshape((1, len(feature_value)), order='A')

        core_if = data_exc.at[i, 'Corefucose']
        if core_if:
            result_core = np.concatenate((result_core, feature_value), axis=0)
        else:
            result_uncore = np.concatenate((result_uncore, feature_value), axis=0)

    result_core = np.delete(result_core, 0, axis=0)
    result_uncore = np.delete(result_uncore, 0, axis=0)
    return result_core, result_uncore



if __name__ == '__main__':
    print(1)
