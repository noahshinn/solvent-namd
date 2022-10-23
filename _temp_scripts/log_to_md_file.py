_LOG_FILE = '../demo/cyclopropenone-15-waters-namd/traj-0.log'
_MD_FILE = 'traj-0.md.xyz'
_NATOMS = 51

def main() -> None:
    with open(_LOG_FILE, 'r') as f:
        data = f.readlines()
        md_data = []
        for i, l in enumerate(data):
            if '&coordinates' in l:
                md_data.extend([f'{_NATOMS}\n\n'])
                md_data.extend(data[i + 2:i + 2 + _NATOMS])    
        with open(_MD_FILE, 'w') as wr_f:
            for l in md_data:
                wr_f.write(l)
    print(f'xyz written to {_MD_FILE}')

if __name__ == '__main__':
    main()
