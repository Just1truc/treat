from sources.argparser import parser

if __name__ == "__main__":
    
    args = parser.parse_args()
    
    from sources.study import study
    from sources.train import train
    
    action_to_function = {
        "study" : study,
        "train" : train
    }
    action_to_function[args.actions](args)