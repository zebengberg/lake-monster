
import datetime
import sys
import argparse
from lake_monster import train

ARG_DICT = {'default': train.generate_default,
            'random': train.generate_random,
            'many': train.run_many_trainings,
            'multi': train.generate_multi,
            'jump': train.generate_jump,
            'clear': train.clear_knowledge,
            'clearall': train.clear_all_knowledge}


def parse_args():
  """Create argparse parser."""

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '-d',
      '--default',
      action='store_true',
      help=ARG_DICT['default'].__doc__,
      required=False)

  parser.add_argument(
      '-r',
      '--random',
      action='store_true',
      help=ARG_DICT['random'].__doc__,
      required=False)

  parser.add_argument(
      '-m',
      '--many',
      action='store_true',
      help=ARG_DICT['many'].__doc__,
      required=False)

  parser.add_argument(
      '--multi',
      action='store_true',
      help=ARG_DICT['multi'].__doc__,
      required=False)

  parser.add_argument(
      '--jump',
      action='store_true',
      help=ARG_DICT['jump'].__doc__,
      required=False)

  parser.add_argument(
      '--clear',
      action='store_true',
      help=ARG_DICT['clear'].__doc__,
      required=False)

  parser.add_argument(
      '--clearall',
      action='store_true',
      help=ARG_DICT['clearall'].__doc__,
      required=False)

  return parser.parse_args(sys.argv[1:])


def main():
  """Parse command line arguments and run training."""

  args = parse_args()
  args = vars(args)  # casting Namespace to dictionary
  for k, v in args.items():
    if v:
      ARG_DICT[k]()
      break


if __name__ == '__main__':
  main()
