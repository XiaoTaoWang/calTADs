# Created on Sat Jan 24 12:46:09 2015

# Author: XiaoTao Wang
# Organization: HuaZhong Agricultural University

import argparse, sys, logging

def getargs():
    ## Construct an ArgumentParser object for command-line arguments
    parser = argparse.ArgumentParser(usage = '%(prog)s <-O output> [options]',
                                     description = 'TAD identification.',
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    # Output
    parser.add_argument('-O', '--output', help = 'Prefix of the generated TAD file, which is '
                        'created under the folder of Hi-C data.')
    
    ## Related to Hi-C data
    parser.add_argument('-p', '--path', default = '.',
                        help = 'Path to the folder with Hi-C data. Support both absolute'
                               ' and relative path.')
    parser.add_argument('-F', '--Format', default = 'NPZ', choices = ['TXT', 'NPZ'],
                        help = 'Format of source data file')
    parser.add_argument('-R', '--resolution', default = 10000, type = int,
                        help = 'Resolution of the binned data')
    parser.add_argument('-T', '--template', default = 'chr%s_chr%s.int',
                        help = 'Template for matching file names using regular expression.'
                        ' Needed when "--Format" is set to be "TXT". Note only within-chromosome'
                        ' data are allowed, and don\'t place inter-chromosome data under the '
                        'folder.')
    parser.add_argument('-C', '--chroms', nargs = '*', default = ['#', 'X'],
                        help = 'Which chromosomes to read. Specially, "#" stands'
                        ' for chromosomes with numerical labels. "--chroms" with zero argument'
                        ' will generate an empty list, in which case all chromosome data will'
                        ' be loaded.')
    parser.add_argument('-c', '--cols', nargs = 3, type = int,
                        help = 'Which 3 columns to read, with 0 being the first. For example,'
                        ' "--cols 1 3 4" will extract the 2nd, 4th and 5th columns. Only '
                        'required when "--Format=TXT".')
    parser.add_argument('--immortal', action = 'store_true',
                        help = 'When specified, a Numpy .npz file will be generated under the same '
                        'folder. This operation will greatly speed up data loading process next'
                        ' time.')
    parser.add_argument('-P', '--prefix', help = 'Prefix of input .npz file name, path not'
                        ' included. Required when "--Format=NPZ".')
    parser.add_argument('-S', '--saveto', help = 'Prefix of output .npz file name, path '
                        'not included. Required with "--immortal".')
    
    ## Related to TAD calculation
    parser.add_argument('-w', '--window', type = int, default = 100,
                        help = '''Window size used in DI (Directionality Index) calculation.
                        It tells how far we need to look at the interaction patterns of a
                        given bin. Unit: RESOLUTION.''')
    parser.add_argument('-m', '--MinSize', type = int, default = 10,
                        help = '''The minimum size of a domain merged from HMM model.''')
    parser.add_argument('--probs', type = float, default = 0.99,
                        help = '''Probability threshold. A domain is acceptable if its median
                        posterior probability surpass this value, although the size is small.''')
    parser.add_argument('--MATLAB', action = 'store_true',
                        help = '''When specified, the HMM model is constructed and fitted using
                        an external MATLAB code.''')
    ## Parse the command-line arguments
    commands = sys.argv[1:]
    if not commands:
        commands.append('-h')
    args = parser.parse_args(commands)
    
    args.Format = args.Format.upper()

    ## Conflicts
    if (args.Format == 'TXT') and (not args.cols):
        parser.print_help()
        parser.exit(1, 'You have to specify "--cols" with "--Format TXT"!')
    if (args.Format == 'NPZ') and (not args.prefix):
        parser.print_help()
        parser.exit(1, 'You have to specify "--prefix" with "--Format NPZ"!')
    if args.immortal and (not args.saveto):
        parser.print_help()
        parser.exit(1, 'You have to specify "--saveto" with "--immortal" flag!')
    
    return args, commands

def run():
     # Parse Arguments
    args, commands = getargs()
    # Improve the performance if you don't want to run it
    if commands[0] not in ['-h', '--help']:
        ## Root Logger Configuration
        logger = logging.getLogger()
        # Logger Level
        logger.setLevel(10)
        console = logging.StreamHandler()
        # Set level for Handlers
        console.setLevel('DEBUG')
        # Customizing Formatter
        formatter = logging.Formatter(fmt = '%(name)-14s %(levelname)-7s @ %(asctime)s: %(message)s',
                                      datefmt = '%m/%d/%y %H:%M:%S')
        
        console.setFormatter(formatter)
        # Add Handlers
        logger.addHandler(console)
        ## Logging for argument setting
        arglist = ['# ARGUMENT LIST:',
                   '# output file name = %s' % args.output,
                   '# data folder = %s' % args.path,
                   '# Hi-C format = %s' % args.Format,
                   '# chromosomes = %s' % args.chroms,
                   '# data resolution = %s' % args.resolution,
                   '# window size = %s' % args.window,
                   '# MinSize = %s' % args.MinSize,
                   '# Probability Threshold = %s' % args.probs,
                   '# Using MATLAB = %s' % args.MATLAB
                   ]
        if args.Format == 'TXT':
            arglist.extend(['# Hi-C file template = %s' % args.template,
                            '# which columns = %s' % args.cols])
        if args.Format == 'NPZ':
            arglist.extend(['# input NPZ prefix = %s' % args.prefix])
        if args.immortal:
            arglist.append('# output NPZ prefix = %s' % args.saveto)
        
        argtxt = '\n'.join(arglist)
        logger.info('\n' + argtxt)
        
        # Interface for Queue Object
        dicts = {'path': args.path, 'Format': args.Format, 'resolution': args.resolution,
                 'template': args.template, 'chroms': args.chroms, 'cols': args.cols,
                 'prefix': args.prefix, 'immortal': args.immortal, 'saveto': args.saveto}
        
        ## Required Modules
        import numpy as np
        from tadlib import analyze
        
        # Queue Object for Data Loading and TAD Identification
        logger.info('Reading Hi-C data ...')
        workInters = analyze.Inters(**dicts)
        
        writeout = open('.'.join([args.output, 'DI', 'txt']), 'w')
        logger.info('Calculating DI for each bin by chromosome ...')
        # Define the structured array type
        tp = np.dtype({'names':['chr','start','end','DI'],
                       'formats':['S2', np.int32, np.int32, np.float]})
        chroms = workInters.labels
        for c in chroms:
            logger.info('Chromosome %s ...', c)
            idata = workInters.data[c]
            L = calDI(idata, args.window, c, tp, args.resolution)
            if args.MATLAB:
                L['chr'] = str(workInters.label2idx[c])
            np.savetxt(writeout, L, delimiter = '\t', fmt = ['%s','%d','%d','%.7e'])
        writeout.flush()
        writeout.close()
        logger.info('Done!')
        
        logger.info('Contructing HMM models for the whole genome ...')
        DIs = np.loadtxt('.'.join([args.output, 'DI', 'txt']), dtype = tp)
        # Define a new structured array format
        tp = np.dtype({'names':['chr','start','end','state','Prob_0','Prob_1','Prob_2'],
                       'formats':['S2', np.int32, np.int32, np.int8, np.float, np.float, np.float]})
        writeout = '.'.join([args.output, 'hmmout', 'txt'])
        if args.MATLAB:
            ## Use the external MATLAB code
            L = MATLABHMM('.'.join([args.output, 'DI', 'txt']), writeout, DIs, tp)
        else:
            ## The HMM model is implemented by Python
            L = GMMHMM(DIs, tp)
            
        np.savetxt(writeout, L, delimiter = '\t', fmt = ['%s','%d','%d','%d','%.7e','%.7e','%.7e'])
        
        logger.info('Post Processing ...')
        states = np.loadtxt('.'.join([args.output, 'hmmout', 'txt']), dtype = tp)
        writeout = open('.'.join([args.output, 'hmmdomain', 'txt']), 'w')
        # Define a new structured array format
        tp = np.dtype({'names':['chr','start','end'], 'formats':['S2', np.int32, np.int32]})
        for c in chroms:
            logger.info('Chromosome %s ...', c)
            if args.MATLAB:
                cstates = states[states['chr']==str(workInters.label2idx[c])]
            else:
                cstates = states[states['chr']==c]
            L = getTAD(cstates, args.MinSize, args.probs, args.resolution, tp)
            L['chr'] = c
            np.savetxt(writeout, L, delimiter = '\t', fmt = ['%s','%d','%d'])
        writeout.flush()
        writeout.close()
        
        logger.info('Done!')
            
def calDI(idata, window, clabel, datatype, Res):
    """
    Calculate DI for each bin. Just input interaction data of one chromosome
    each time.
    """
    import numpy as np
    # Perform filtering according to window size
    mask = ((idata['bin2'] - idata['bin1']) <= window) & (idata['bin2'] != idata['bin1'])
    idata = idata[mask]
    # Create a structured array for output
    Rbound = idata['bin2'].max()
    Len = Rbound + 1
    output = np.zeros(Len, dtype = datatype)
    output['chr']  = clabel
    output['start'] = np.arange(Len) * Res
    output['end'] = np.arange(1, Len + 1) * Res
    # Downstream
    downs = np.bincount(idata['bin1'], weights = idata['IF'])
    # Upstream
    ups = np.bincount(idata['bin2'], weights = idata['IF'])
    ## Correct for length
    cdowns = np.zeros(Len)
    cdowns[:downs.size] = downs
    cups = np.zeros(Len)
    cups[(Len-ups.size):] = ups
    ## Formula
    numerator = cdowns - cups
    denominator_1 = cdowns + cups
    denominator_2 = numerator.copy()
    denominator_1[denominator_1==0] = 1
    denominator_2[denominator_2==0] = 1
    DI = numerator**3 / np.abs(denominator_2) / denominator_1
    
    # Update structured array
    output['DI'] = DI
    
    return output

def MATLABHMM(filename, output, DI, datatype):
    ## Necessary Modules
    import os
    import numpy as np
    
    ## Call external programs
    logging.info('Call MATLAB functions ...')
    logfile = os.path.basename(filename).replace('.DI.txt', '.HMM.log')
    errorfile = os.path.basename(filename).replace('.DI.txt', '.HMM.err')
    
    command = ' '.join(['matlab','-nojvm','-nodisplay','-nosplash','-nodesktop',
                        '-r',''.join(['"HMM_calls(', '\'', filename, '\'', ',', '\'', output,
                        '\'',')"']), ''.join(['1>',logfile]), ''.join(['2>',errorfile])])
    
    os.system(command)
    
    logging.info('Done!')
    
    # Correction
    logging.info('Re-delegate state index according to DI ...')
    
    Raw = np.loadtxt(output)
    hidden_states = Raw[:,1] - 1
    Means = np.zeros(3)
    for i in range(3):
        temp = DI['DI'][hidden_states==i]
        Means[i] = temp.mean()
    s = np.argsort(Means) # The Key
    
    # Create the output array
    output = np.zeros(DI.size, dtype = datatype)
    output['chr'] = DI['chr']
    output['start'] = DI['start']
    output['end'] = DI['end']
    output['state'][hidden_states==s[0]] = 0 # Upstream Bias
    output['state'][hidden_states==s[1]] = 1 # No Bias
    output['state'][hidden_states==s[2]] = 2 # Downstream Bias
    output['Prob_0'] = Raw[:,s[0]+2]
    output['Prob_1'] = Raw[:,s[1]+2]
    output['Prob_2'] = Raw[:,s[2]+2]
    
    logging.info('Finished!')
    
    return output
    
    
def GMMHMM(DI, datatype):
    # Necessary Modules
    from hmmlearn import hmm
    import numpy as np
    
    # Parameters
    maxM = 20
    components = 3
    O = 1
    # Data
    train_set = DI['DI']
    train_set = train_set.reshape(DI.size, O)
    
    # Records
    aic = np.zeros(maxM)
    Ks = np.zeros(maxM, dtype=int)
    states_pool = np.zeros((DI.size, maxM), dtype=int)
    probs = {}
    
    for i in range(1, maxM + 1):
        logging.info('Mixture Number: %s', i)
        
        K = 0 # Number of Parameters for AIC Calculation
        # Gaussian Mixture Model Parameters
        K += (components * (i * (O*2 + 1) - 1))
        # HMM Model
        model = hmm.GMMHMM(n_components = components, n_mix = i, n_iter = 2000,
                           thresh = 1e-10, covariance_type = 'full')
        # HMM Parameters
        prior = model.startprob_
        transmat = model.transmat_
        K += np.prod(prior.shape)
        K += np.prod(transmat.shape)
        
        # Training
        logging.info('Training ...')
        model.fit([train_set])
        logging.info('Done!')
        
        logL = model.score(train_set)
        hidden_states = model.predict(train_set)
        states_pool[:, i-1] = hidden_states
        probs[i-1] = model.predict_proba(train_set)
        aic[i-1] = -2*logL + 2*K + 2*K*(K + 1) / (DI.size - K - 1)
        Ks[i-1] = K
        
        logging.info('Log likelihood: %s, AIC value: %s, Parameter Number: %s',
                     logL, aic[i-1], K)
        
        
    # Apply AIC
    order = np.int(np.floor(np.log10(abs(aic.min())))) - 1
    div = np.power(10, order)
    
    # Relative probability
    for i in range(maxM):
        p_aic = np.exp((aic.min() - aic[i]) / (div * 2))
        if p_aic >= 0.9:
            idx = i
            break
    
    # Selected states
    hidden_states = states_pool[:,idx]
    # Corresponding posterior probability for each state
    posterior_probs = probs[idx]
    
    logging.info('The Best Model:')
    logging.info('The minumum AIC: %s', aic.min())
    logging.info('AIC of selected model: %s', aic[idx])
    logging.info('Relative Probability: %s', p_aic)
    logging.info('Model Index: %s', idx + 1)
    logging.info('Number of parameters: %s', Ks[idx])
    
    
    logging.info('Re-delegate state index according to DI ...')
    Means = np.zeros(components)
    for i in range(components):
        temp = train_set[hidden_states==i]
        Means[i] = temp.mean()
    s = np.argsort(Means) # The Key
    
    # Create the output array
    output = np.zeros(DI.size, dtype = datatype)
    output['chr'] = DI['chr']
    output['start'] = DI['start']
    output['end'] = DI['end']
    output['state'][hidden_states==s[0]] = 0 # Upstream Bias
    output['state'][hidden_states==s[1]] = 1 # No Bias
    output['state'][hidden_states==s[2]] = 2 # Downstream Bias
    output['Prob_0'] = posterior_probs[:,s[0]]
    output['Prob_1'] = posterior_probs[:,s[1]]
    output['Prob_2'] = posterior_probs[:,s[2]]
    
    logging.info('Finished!')
    
    return output

def getTAD(states, MinSize, probs, Res, datatype):
    """
    Estimate the median posterior probability of a region(a stretch of same
    state). We believe in a region only if it has a median posterior
    probability >= 0.99, or its size surpass 100 Kb.
    
    TADs always begin with a single downstream biased state, and end with
    a last HMM upstream biased state.
    """
    # Necessary Modules
    import numpy as np
    
    # Stretch consecutive same state  -->  Region
    mediate = []
    start = states[0]['start']
    end = states[0]['end']
    cs = states[0]['state'] # Current State
    prob_pool = [states[0][4+cs]]
    
    for i in xrange(1, states.size):
        line = states[i]
        if line['state'] != cs:
            mediate.append([start, end, cs, np.median(prob_pool)])
            start = line['start']
            end = line['end']
            cs = line['state']
            prob_pool = [line[4+cs]]
        else:
            end = line['end']
            prob_pool.append(line[4+cs])
    
    dawn = []
    # Calibrate the first and the last line
    if (mediate[0][1] - mediate[0][0]) <= 3*Res:
        mediate[0][2] = mediate[1][2]
    if (mediate[-1][1] - mediate[-1][0]) <= 3*Res:
        mediate[-1][2] = mediate[-2][2]
    
    dawn.append([mediate[0][0], mediate[0][1], mediate[0][2]])
    # Two criteria
    for i in xrange(1, len(mediate)-1):
        temp = mediate[i]
        if ((temp[1] - temp[0]) >= Res*MinSize) or (temp[-1] > probs):
            dawn.append([temp[0], temp[1], temp[2]])
        else:
            Previous = mediate[i-1]
            Next = mediate[i+1]
            if Previous[2] == Next[2]:
                dawn.append([temp[0], temp[1], Previous[2]])
            else:
                dawn.append([temp[0], temp[1], 1])
    
    dawn.append([mediate[-1][0], mediate[-1][1], mediate[-1][2]])
    
    ## Infer TADs
    preTADs = []
    # Artificial Chromosome Size
    genome_size = dawn[-1][1]
    temp = []
    for i in xrange(len(dawn)):
        start = dawn[i][0]
        end = dawn[i][1]
        state = dawn[i][2]
        if i == 0:
            pre_state = state
            pre_end = end
            continue
        if state != pre_state:
            if pre_state == 1:
                temp.append(start)
            if state == 1:
                temp.extend([pre_end, pre_state])
                preTADs.append(temp)
                temp = []
            if (pre_state != 1) and (state != 1):
                temp.extend([pre_end, pre_state])
                preTADs.append(temp)
                temp = [start]
        
        pre_end = end
        pre_state = state
    
    if pre_state != 1:
        temp.extend([genome_size, pre_state])
        preTADs.append(temp)
    
    TADs = []
    pre_state = -1
    Chrom = states[0]['chr']
    temp = [Chrom]
    for i in xrange(len(preTADs)):
        if pre_state == -1:
            if (preTADs[i][-1] != 2) or (len(preTADs[i]) < 3):
                continue
            
        start = preTADs[i][0]
        end = preTADs[i][1]
        state = preTADs[i][2]
        
        if state != pre_state:
            if (state == 2) and (pre_state == -1):
                temp.append(start)
            if (state == 2) and (pre_state == 0):
                temp.append(pre_end)
                TADs.append(tuple(temp))
                temp = [Chrom, start]
        
        pre_state = state
        pre_end = end
        
    if (pre_state == 0) and (len(temp) == 2):
        temp.append(pre_end)
        TADs.append(tuple(temp))
    
    TADs = np.array(TADs, dtype = datatype)
    
    return TADs
            

if __name__ == '__main__':
    run()
